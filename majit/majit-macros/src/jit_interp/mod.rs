//! `#[jit_interp]` proc macro implementation.
//!
//! Transforms an interpreter mainloop function into a JIT-enabled version by:
//! 1. Generating `trace_instruction` from the match dispatch
//! 2. Generating `JitState` types and impl
//! 3. Replacing `jit_merge_point!()` / `can_enter_jit!()` markers

pub(crate) mod call_policy_byte;
mod classify;
mod codegen_state;
pub(crate) mod codegen_trace;
mod green_type_tag;
pub(crate) mod jitcode_lower;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Expr, Ident, ItemFn, LitBool, Path, Token, braced, bracketed,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Parsed configuration from `#[jit_interp(...)]` attributes.
///
/// ## Helper discovery
///
/// Helpers (functions called from traced match arms) can be declared in
/// three ways, from most explicit to most concise:
///
/// 1. **`calls = { helper_a, helper_b => residual_int, ... }`**
///    Brace-delimited list with optional per-helper policy overrides.
///
/// 2. **`helpers = [helper_a, helper_b, helper_c]`**
///    Bracket-delimited shorthand — all helpers use auto-inferred policy.
///    Equivalent to listing each in `calls = { ... }` without a `=>` override.
///    Can be combined with `calls` for helpers that need explicit policies.
///
/// 3. **`auto_calls = true`**
///    Infer helper policies from sidecar `#[elidable]` / `#[dont_look_inside]`
///    / `#[jit_inline]` attributes on every call site in the traced arms.
///    Value-call inference is currently limited to helpers whose result bank
///    is statically int-shaped; ref/float-return helpers still need explicit
///    `calls = { helper => ... }` overrides such as `inline_ref`,
///    `inline_float`, `residual_ref_wrapped`, or `residual_float_wrapped`.
///
/// ### Module-level discovery
///
/// For automatic helper discovery, use `#[jit_module]` on the enclosing
/// `mod` block. It scans all items for JIT-annotated functions and
/// generates hidden registry constants (`__MAJIT_DISCOVERED_HELPERS`,
/// `__MAJIT_HELPER_POLICIES`). Alternatively, use `helpers` or `calls`
/// to explicitly list the functions that need JIT integration.
pub struct JitInterpConfig {
    pub trace_cfg: Option<syn::Meta>,
    /// The interpreter state type (e.g., `InterpState`).
    pub state_type: Ident,
    /// The environment type (e.g., `Program`).
    pub env_type: Ident,
    /// Interpreter I/O function → JIT shim function mapping.
    pub io_shims: Vec<(Path, Ident)>,
    /// Interpreter function call policies for helper calls.
    /// Populated from both `calls = { ... }` and `helpers = [...]`.
    pub calls: Vec<CallEntry>,
    /// Whether direct helper calls should be auto-inferred from sidecar metadata.
    pub auto_calls: bool,
    /// Optional structured green-key expressions for marker rewrite.
    pub greens: Vec<Expr>,
    /// Whether the attribute spelled a `greens` key at all, as opposed to
    /// declaring `greens = []`.
    ///
    /// `greens` alone cannot answer that: both spellings produce an empty
    /// `Vec`, and the merge-point refusal needs to tell them apart. An
    /// omitted key is an author who has not considered greens; `greens = []`
    /// is an author who has, and says so — the encoded merge point is
    /// identical either way, so this is the only place the difference
    /// survives.
    pub greens_declared: bool,
    /// Slice (audit Issue #6) — explicit red declarations for the
    /// dispatch JitCode `BC_JIT_MERGE_POINT` payload.  RPython
    /// `jtransform.py make_three_lists(op.args[2+num_green_args:])`
    /// derives reds from the marker call's tail args; pyre's marker is
    /// stateless (no tail args), so consumers declare the reds via
    /// this config slot instead.  Empty = use the default candidate
    /// list `[program, pc(+ optional vable)]` with declared greens
    /// filtered out (the pre-Issue-#6 pyre behavior).
    pub reds: Vec<Expr>,
    /// Per-green type tag from `greens = [name: tag]` syntax.
    /// Lockstep with [`Self::greens`] (same length, same order).  `None`
    /// at a given index means the green flows through the
    /// `<_ as GreenAsI64>::__green_repr` trait dispatch unchanged;
    /// `Some(tag)` overrides the dispatch with an explicit
    /// `(value_bits, GreenType::<tag>)` pair so str / unicode greens route
    /// through the hardcoded `default_str_eq` / `default_str_hash` /
    /// `default_unicode_hash` in `majit-ir/src/value.rs`
    /// (`warmstate.py equal_whatever lltype.Ptr STR/UNICODE` parity, no
    /// frontend override).
    pub green_type_tags: Vec<Option<green_type_tag::GreenTypeTag>>,
    /// Virtualizable frame field declaration.
    ///
    /// RPython equivalent: jtransform.py's `is_virtualizable_getset()`.
    /// When set, the proc macro rewrites field accesses on the virtualizable
    /// variable to use TraceCtx vable_* methods instead of heap operations.
    pub virtualizable_decl: Option<VirtualizableDecl>,
    /// State field declarations for register/tape machines.
    ///
    /// When set, the macro tracks state struct fields as JIT-managed values
    /// instead of requiring a storage pool. Enables `state.field` and
    /// `state.array[index]` patterns in match arms.
    pub state_fields: Option<StateFieldsConfig>,
    /// Optional recovery method path called after compiled loop guard failure.
    /// The method is called on `&mut self` (the state type) to re-derive
    /// tracked state fields from shared mutable state after the compiled
    /// loop's residual calls have mutated it.
    pub recover: Option<Path>,
    /// Optional concrete fallback for `recursive_portal_call!` (a recursive
    /// portal re-entry).  The function is the runtime analog of the portal
    /// runner (`jd.portal_runner_ptr`, call.py): the transformed
    /// (concrete) function calls it with the macro greens forwarded
    /// positionally (jitdriver declaration order), while the dispatch JitCode
    /// emits `BC_RECURSIVE_CALL_*`.  Required iff the body uses
    /// `recursive_portal_call!`.
    pub recursive_entry: Option<Path>,
    /// Residual helpers that mutate a `state.<ref_scalar>.<field>` through
    /// opaque host code, declared as `residual_writes = { <ref_scalar>.<field>
    /// [@ StructType] => [helpers] }`.  `@ StructType` adds a nominal-layout
    /// alias for a type-punned ref scalar.  Drives the write-set `EffectInfo` the lowering attaches
    /// so the optimizer invalidates the matching cached `getfield_gc_i` after
    /// the call.  Empty for interpreters with no residual field mutators.
    pub residual_writes: Vec<ResidualWriteEntry>,
    /// `ref(T)` state scalars that are bases of a contiguous raw-pointer array
    /// (`[*mut U; N]` at offset 0 of `T`), declared as
    /// `pool_arrays = { <ref> => <getter> [-> ElementType], ... }`.  The
    /// indexing marker call `<getter>(state.<ref>, <int>)` lowers to
    /// `getarrayitem_gc_r` (a
    /// re-producible heap read) instead of an opaque residual CALL_R, so the
    /// loaded element re-derives from the index each loop entry and the short
    /// preamble can re-emit it.  Selection is keyed on the `getter` function
    /// identity (not the arg shape alone).  Empty for interpreters with no
    /// pool-array indexing.
    pub pool_arrays: Vec<PoolArrayEntry>,
    /// Struct field type declarations for ref-kind field access through
    /// `ref(T)` state scalars and non-state ref bindings.  Declared as
    /// `ref_fields = { Struct::field => PointeeType, ... }`.  When the
    /// lowerer encounters `state.<ref_scalar>.<field>` or `<binding>.<field>`
    /// and `(Struct, field)` is listed here, it emits `getfield_gc_r` /
    /// `setfield_gc_r` (ref-kind) instead of the default `_gc_i` (int-kind).
    /// The `PointeeType` is recorded so subsequent field access on the
    /// returned ref can resolve its struct layout.
    pub ref_fields: Vec<RefFieldEntry>,
    /// Array base-pointer field declarations, `array_fields = { Struct::field
    /// => ElementType, ... }`.  See [`ArrayFieldEntry`].
    pub array_fields: Vec<ArrayFieldEntry>,
    /// Sub-word integer field declarations, `int_fields = { Struct::field =>
    /// u32, ... }`.  A field access lowers to `getfield_gc_i` /
    /// `setfield_gc_i` either way; what this adds is the field's real width
    /// and signedness, which the descr otherwise reports as a full signed
    /// machine word.  `descr.py get_field_descr` takes both from
    /// `FIELDTYPE`, and `intbounds.py` narrows a load's range only when
    /// `descr.is_integer_bounded()` — true exactly for a sub-word integer
    /// field.  Undeclared fields keep the machine-word default, so this is
    /// opt-in per field; a declaration that disagrees with the Rust struct
    /// fails to compile (see the generated `const _` check).
    pub int_fields: Vec<IntFieldEntry>,
    /// Struct type annotations for ref-returning call results.
    /// `call_returns = { func_path => StructType, ... }`.  When a
    /// `residual_ref` (or similar ref-returning) call result is bound,
    /// the lowerer sets `Binding.struct_type` to the declared type so
    /// subsequent `result.field` access resolves through `ref_fields`.
    pub call_returns: Vec<(Path, Path)>,
    /// Struct literal → concrete allocator mapping for the concrete path.
    /// `struct_allocs = { StructType => allocator_func, ... }`.  When the
    /// concrete-path `RefFieldRewriter` encounters a struct literal
    /// `StructType { f0: v0, f1: v1 }`, it rewrites the literal to
    /// `allocator_func(v0, v1)`.  The JIT path already handles struct
    /// literals natively via `lower_struct_value` (New + SetfieldGc).
    pub struct_allocs: Vec<(Path, Path)>,
    /// Structs whose `New` allocation should use the headerless nursery opcode.
    /// `headerless_structs = { StructType, ... }`.
    pub headerless_structs: Vec<Path>,
    /// Leading-substructure declarations, `inlined_prefix = { Outer::field =>
    /// Base, ... }`.  See [`InlinedPrefixEntry`].  Empty by default, in which
    /// case every access resolves against the struct it is spelled with and the
    /// expansion is unchanged.
    pub inlined_prefix: Vec<InlinedPrefixEntry>,
    /// Pure function → native IR integer binop aliases.
    /// `native_int_binops = { val_add => IntAdd, ... }`.  When the JIT-path
    /// lowerer encounters a call whose path matches a key, it emits the named
    /// IR opcode (e.g. `IntAdd`) directly instead of routing through the
    /// call-policy machinery.  The concrete path calls the function normally.
    /// In RPython, `int + int` is lowered to `int_add` at rtype time
    /// (rint.py `_rtype_template`), so the JIT codewriter never sees a
    /// call.  In pyre's LLBC tracer, the equivalent Rust function call
    /// (e.g. `val_add`) is still visible, so this config rewrites it to a
    /// native IR binop at codewriter time.
    pub native_int_binops: Vec<(Path, Ident)>,
    /// native_tag_small = { jit_retag_small } — unary fns lowered to (x<<1)|1 native IR.
    pub native_tag_small: Vec<Path>,
    /// Pure function → its own argument.  `native_identity = { unwrap_word }`.
    /// A call whose path matches one of these is replaced by the binding of
    /// its single argument, so the trace never sees a call at all.  What opens
    /// a helper this small upstream is the inliner, not `same_as` handling:
    /// `warmspot.WarmRunnerDesc.__init__` runs `prejit_optimizations` →
    /// `inline.auto_inline_graphs` → `BaseInliner.do_inline`, then
    /// `BaseInliner.cleanup` → `simplify.cleanup_graph`, all before
    /// `make_jitcodes()`, and that leaves no residue for the codewriter to
    /// drop.  This front end lowers a `syn` AST straight to a `JitCode` with
    /// no flow-graph stage for an inliner to run in, so the declaration is
    /// that pass relocated to the only stage there is.
    /// Declaring a function whose argument and result do not share a register
    /// class is a miscompile — the argument's binding is handed straight to
    /// the consumer.
    pub native_identity: Vec<Path>,
    /// Opt-in: route pure forward-advancing dispatch arms (those whose body
    /// only does work then `pc += N`, with no back-edge / `can_enter_jit!` /
    /// early return) through the per-arm sub-JitCode path with a pc-returning
    /// `inline_call_<types>_i`, instead of force-inlining them into the
    /// dispatch JitCode.  This keeps the dispatch JitCode's register/const
    /// footprint small (the heavy arm bodies move into their own sub-JitCodes,
    /// each with its own ≤256 budget) — mirrors `next_instr = self.OPCODE(...)`.
    /// Off by default, so kernels that do not set it are byte-identical.
    pub split_dispatch: bool,
    /// Opt-in: lower the top-level opcode dispatch discrimination as one
    /// `switch/id` op instead of a `goto_if_not_int_eq` guard chain. Off by
    /// default so existing interpreters keep byte-identical dispatch JitCode.
    pub switch_dispatch: bool,
}

/// Virtualizable frame field declaration for `#[jit_interp]`.
///
/// RPython equivalent: VirtualizableInfo from virtualizable.py, combined
/// with jtransform.py's field-to-descriptor mapping.
///
/// Syntax in attribute:
/// ```ignore
/// virtualizable_fields = {
///     var: frame,
///     token_offset: PYFRAME_VABLE_TOKEN_OFFSET,
///     fields: { next_instr: int @ NEXT_INSTR_OFFSET, ... },
///     arrays: {
///         locals_w: ref @ LOCALS_OFFSET,
///         stack: int @ (DATA_PTR_OFFSET + SLOT_OFFSET) {
///             ptr_offset: 0,
///             length_offset: LENGTH_OFFSET_MINUS_DATA_PTR,
///             items_offset: 0,
///         },
///     },
/// }
/// ```
pub struct VirtualizableDecl {
    /// Expression for the virtualizable variable in the mainloop body.
    pub var_name: Ident,
    /// Constant path for the vable_token field offset.
    pub token_offset: Path,
    /// Static fields: name, type (int/ref/float), byte offset constant.
    pub fields: Vec<VableFieldDecl>,
    /// Array fields: name, item type (int/ref/float), byte offset constant.
    pub arrays: Vec<VableArrayDecl>,
}

/// A single virtualizable static field declaration.
pub struct VableFieldDecl {
    /// Field name as it appears in the struct.
    pub name: Ident,
    /// Field type: `int`, `ref`, or `float`.
    pub field_type: Ident,
    /// Byte offset expression (e.g., `PYFRAME_NEXT_INSTR_OFFSET`,
    /// `STORAGEPOOL_LENGTHS_OFFSET + 8`).
    pub offset: Expr,
}

/// A single virtualizable array field declaration.
pub struct VableArrayDecl {
    /// Array field name as it appears in the struct.
    pub name: Ident,
    /// Item type: `int`, `ref`, or `float`.
    pub item_type: Ident,
    /// Physical layout of the array field inside the virtualizable object.
    pub layout: VableArrayLayoutDecl,
}

/// Layout description for a virtualizable array field.
#[allow(clippy::large_enum_variant)]
pub enum VableArrayLayoutDecl {
    /// Direct pointer field: the virtualizable stores a pointer to the
    /// array object. `length_offset` and `items_offset` describe where,
    /// relative to that pointer, the length prefix and items begin.
    /// Matches RPython `Ptr(GcArray(T))`: `length_offset = 0`,
    /// `items_offset = size_of::<usize>()` (8 on 64-bit).
    ///
    /// `None` defaults to `0` for back-compat with pointer-to-items shapes
    /// where items sit directly at the field pointer.
    Direct {
        field_offset: Expr,
        length_offset: Option<Expr>,
        items_offset: Option<Expr>,
    },
    /// Embedded container layout: pointer/length live in sibling fields or
    /// an inline container relative to the declared field offset.
    Embedded {
        field_offset: Expr,
        ptr_offset: Expr,
        length_offset: Expr,
        items_offset: Expr,
    },
}

/// State field declaration for register/tape machines.
///
/// Syntax: `state_fields = { a: int, x: float, regs: [int], ... }`
pub struct StateFieldsConfig {
    pub fields: Vec<StateFieldDecl>,
}

/// A single state field declaration.
pub struct StateFieldDecl {
    pub name: Ident,
    pub kind: StateFieldKind,
}

/// Whether a state field is a scalar, array, virtualizable array, or
/// an opaque pass-through (RPython parity: a field that lives on the state
/// struct but is invisible to the JIT — accessed only through explicit
/// interpreter code or via raw-pointer handles. Used for `Storage`-like
/// pools with polymorphic dispatch that can not be flattened as ints).
pub enum StateFieldKind {
    /// Scalar value.
    ///
    /// Syntax: `a: int` (default i64 storage), `x: float` (default f64
    /// storage), or `a: int(usize)` /
    /// `a: int(i32)` to declare a different Rust storage type. RPython
    /// parity: `lltype.Signed` is i64 word-sized, `lltype.Unsigned` is
    /// usize word-sized — both render to a single Int register in IR
    /// but the user-visible struct field carries the declared type. The
    /// macro emits `as i64` / `as <type>` casts at the JIT boundary so
    /// IR sees a uniform i64 while the interpreter keeps natural Rust
    /// types (e.g. `selected: usize`, `stacksize: i32`).
    Scalar {
        ir_type: Ident,
        /// `None` ⇒ Rust storage type matches `ir_type` (default i64
        /// for `int`). `Some(path)` ⇒ explicit override from
        /// `int(<path>)` syntax.
        rust_type: Option<syn::Path>,
    },
    /// Array value (e.g., `regs: [int]`) — items-pointer then length.
    /// `rlist.py` `rtype_getitem` uses `dum_nocheck` for a plain `l[i]`;
    /// `ll_getitem_nonneg`'s only check is `ll_assert`, which translation
    /// removes. `jtransform.py` `rewrite_op_getarrayitem` raw arm
    /// (`ARRAY._gckind == 'raw'`) emits `getarrayitem_raw_<kind>`.
    Array(Ident),
    /// Virtualizable array (e.g., `tape: [int; virt]`) — one vable red.
    /// Cells live in `virtualizable_boxes` (`getarrayitem_vable_*`).
    VirtArray(Ident),
    /// Opaque pass-through (e.g., `storage: opaque(Storage)`) — the field
    /// keeps its declared type on the state struct but is NOT enumerated
    /// as a JIT inputarg, fail_arg, or sym slot. The codegen skips it
    /// entirely; the interpreter is responsible for accessing it directly.
    /// Used to carry pools/handles whose internal layout is not flat ints
    /// (e.g. polymorphic `Storage`). Pair with `opaque(Type)` raw-pointer
    /// handles passed via additional `int` scalars when the JIT needs to
    /// touch the underlying memory through `jit_promote!` + raw-load IR.
    #[allow(dead_code)]
    Opaque(syn::Path),
    /// Ref-typed scalar (e.g., `sel: ref(Stack)`) — a genuine `InputArgRef`
    /// threaded through the ref register bank (`registers_r`) rather than the
    /// int bank.  Storage on the state struct is a `usize` carrier (raw
    /// GcRef / pointer bits); the `ref(T)` type path `T` names the heap class
    /// the value points at.  Routing it through the ref bank lets the
    /// optimizer heap-cache `getfield_gc` off it (a stable ref box), unlike
    /// the cosmetic `input_arg_ref` virt-array pointer which flows through the
    /// int bank.
    #[allow(dead_code)]
    Ref(syn::Path),
    /// RPython `str` red (`rstr.STR`). One ref inputarg, same bank as
    /// `ref(T)`. `ll_strlen` / `ll_strgetitem` on it record `Strlen` /
    /// `Strgetitem` (`jtransform.py` `rewrite_op_getinteriorfield`).
    Str,
}

/// Reject a path declared in two of the mutually exclusive call vocabularies.
///
/// `lower_value.rs`'s call arm tries `native_int_binops`, `native_tag_small`
/// and `native_identity` in that order and only then the `calls` policy
/// machinery, and each match returns immediately. A path in two of them
/// therefore takes the earlier route and the later declaration is dead — with
/// nothing downstream able to say so, because by that point only the winning
/// route survives. The declaration order below is the lowerer's, so the
/// vocabulary reported first is the one that actually applies.
///
/// Two entries inside a single vocabulary are left alone: `calls` is filled
/// from both `calls = { ... }` and `helpers = [ ... ]`, where naming a
/// function in both is how a policy is attached to a discovered helper.
pub(crate) fn reject_overlapping_call_vocabularies(
    calls: &[CallEntry],
    native_int_binops: &[(Path, Ident)],
    native_tag_small: &[Path],
    native_identity: &[Path],
) -> syn::Result<()> {
    let vocabularies: [(&str, Vec<&Path>); 4] = [
        (
            "native_int_binops",
            native_int_binops.iter().map(|(path, _)| path).collect(),
        ),
        ("native_tag_small", native_tag_small.iter().collect()),
        ("native_identity", native_identity.iter().collect()),
        ("calls", calls.iter().map(|entry| &entry.path).collect()),
    ];

    let mut declared: Vec<(Vec<String>, &str)> = Vec::new();
    for (vocabulary, paths) in &vocabularies {
        for path in paths {
            let segments = jitcode_lower::canonical_path_segments(path);
            if let Some((_, winner)) = declared
                .iter()
                .find(|(other, owner)| *other == segments && owner != vocabulary)
            {
                return Err(syn::Error::new_spanned(
                    path,
                    format!(
                        "`{name}` is declared in both `{winner}` and `{vocabulary}`; \
                         the lowering takes the `{winner}` route and the \
                         `{vocabulary}` declaration never applies. Declare it in \
                         exactly one.",
                        name = segments.join("::"),
                    ),
                ));
            }
            declared.push((segments, vocabulary));
        }
    }
    Ok(())
}

/// One entry in the `calls = { ... }` / `helpers = [ ... ]` map.
#[derive(Clone)]
pub struct CallEntry {
    pub path: Path,
    pub policy: Option<CallPolicyKind>,
}

/// One entry in the `residual_writes = { <ref_scalar>.<field> [@ StructType] => [helpers] }`
/// map.  Each listed residual helper mutates `<ref_scalar>.<field>` through
/// opaque host code; the lowering attaches a write-set `EffectInfo` naming that
/// field so `OptHeap::force_from_effectinfo` invalidates the cached
/// `getfield_gc_i` after the call (the residual analogue of a traced mutator's
/// in-trace `setfield_gc` write barrier).  `ref_scalar` is resolved against
/// `state_ref_scalars` to recover the struct `Path` for `offset_of!` +
/// `struct_type_id`.
#[derive(Clone)]
pub struct ResidualWriteEntry {
    pub ref_scalar: Ident,
    pub field: Ident,
    /// Optional nominal struct layout alias.  This is needed when the raw
    /// pointer carried by `ref_scalar` type-puns several repr(C) layouts whose
    /// shared field offsets nevertheless have distinct field-descr identities.
    pub struct_type: Option<Path>,
    /// Set by the `[]` suffix (`selected_ref.data[] => [...]`): the helpers
    /// mutate the ELEMENTS of the array `<field>` points at, not the field
    /// itself.  The two are independent claims — a residual that reallocates a
    /// buffer writes both, and one that only stores through it writes only the
    /// elements — so a declaration of each is written separately.
    pub writes_elements: bool,
    pub helpers: Vec<Path>,
}

/// A `ref(T)` state scalar that is the base of a contiguous raw-pointer array,
/// paired with the marker `getter` function whose call indexes it.  Declared as
/// `pool_arrays = { <ref>.<items>[<len>] => <getter> [-> ElementType], ... }`.
/// The lowering recognizes a pool read only when BOTH the call's function path
/// matches `getter` AND arg0 is `state.<base>` — operation identity, not arg
/// shape alone, so an unrelated helper that happens to take the same
/// `(state.<base>, int)` shape is not miscompiled into a `getarrayitem_gc_r`.
#[derive(Clone)]
pub struct PoolArrayEntry {
    pub base: Ident,
    pub getter: Path,
    /// The array field inside the base struct, e.g. `pools` in
    /// `Storage { pools: [*mut T; N], .. }`.  Its `offset_of!` is what the
    /// element read adds to the base pointer.  Naming it is what leaves the
    /// struct free to hold anything else: undeclared, the read has to assume a
    /// layout and the consumer has to build its struct to match, with the
    /// agreement written on neither side in anything the compiler checks.
    pub items_field: Ident,
    /// A length word ahead of the items, declared as `<items>[<len>]`.  Absent
    /// for a plain fixed-size array, which has no length in memory; present
    /// when the consumer keeps one, which buys the `len > index` bound the
    /// short preamble re-establishes on each loop entry.
    pub len_field: Option<Ident>,
    /// The concrete element type each array slot points at (`items: [*mut T; N]`),
    /// declared as `... => <getter> -> T`.  Supplies `struct_type` for the
    /// getter's ref binding so field access on a LOCAL `let x = <getter>(...)`
    /// resolves the layout (state-field refs get this from their decl instead).
    pub element_type: Option<Path>,
}

/// One entry in `ref_fields = { Struct::field => PointeeType, ... }`.
/// Declares that `Struct.field` is a ref-kind (pointer) field whose pointee
/// is `PointeeType`.  The lowerer uses this to emit `getfield_gc_r` /
/// `setfield_gc_r` for the field access, and tags the resulting ref binding
/// with `pointee_type` so subsequent field access on it resolves the layout.
#[derive(Clone)]
pub struct RefFieldEntry {
    /// The struct that owns the field (e.g. `Stack`, `Node`).
    pub struct_type: Path,
    /// The field name within the struct (e.g. `head`, `next`).
    pub field: Ident,
    /// The struct the pointer points to (e.g. `Node`).
    pub pointee_type: Path,
}

/// One entry in `inlined_prefix = { Outer::field => Base, ... }`.
///
/// Declares that `Outer.field` is an inlined substructure of type `Base`
/// occupying the leading bytes of `Outer` — the relationship `rclass.py:548`
/// builds into every instance struct as
/// `MkStruct(name, ('super', rbase.object_type), *own_fields)`.
///
/// What it buys is descr identity. A field the outer struct does not itself
/// declare is *owned* by the base, so an access reaching it is re-expressed
/// against the base before the field key and the type id are built.
/// `rclass.py InstanceRepr.getfield` does the same by recursing to
/// `self.rbase` with `force_cast=True`, which leaves `jtransform.py:881`
/// reading the descr off the declaring struct in every case — so one physical
/// field has one descr for a whole family of outer structs, however many of
/// them exist.
///
/// The base must be the leading field. `lltype.py _first_struct` admits an inlined
/// substructure only at `_names[0]`, and the offset arithmetic depends on it:
/// a redirected access keeps the outer pointer in its register and offsets by
/// the field's position within the *base*, which names the right word only when
/// the base starts at zero. Each emit site therefore also emits a
/// `const _: () = assert!(offset_of!(Outer, field) == 0)`.
#[derive(Clone)]
pub struct InlinedPrefixEntry {
    /// The struct that embeds the base as its leading field.
    pub outer: Path,
    /// The field of `outer` holding the embedded base.
    pub base_field: Ident,
    /// The embedded leading substructure's own type, which declares the
    /// fields the outer struct inherits.
    pub base: Path,
}

/// One entry in `array_fields = { Struct::field => ElementType, ... }`.
///
/// Declares that `Struct.field` holds the base pointer of a contiguous
/// array of `ElementType`, so `base.field[i]` lowers to a `getfield_gc_r`
/// for the buffer pointer followed by `get/setarrayitem_gc_i` on it.
///
/// `ElementType in Header` is the same declaration for pointer elements
/// that begin at `Header`'s `items` field. The load is `getarrayitem_gc_r`
/// with that field's offset as `base_size`. Integer elements stay on
/// `get/setarrayitem_gc_i` and have no header.
///
/// This is the non-virtualizable array vocabulary: the elements do NOT ride
/// a guard's `vable_array` resume section, so the snapshot stays O(1) in the
/// array's length. A `[..; virt]` state field is the other choice and is only
/// correct for per-frame state whose length is bounded by the green key.
///
/// The buffer carries no object header the length op can read, so its length
/// is NOT readable with `arraylen_gc`; bound an index against a sibling
/// `int_fields` length/offset field instead.
#[derive(Clone)]
pub struct ArrayFieldEntry {
    /// The struct that owns the field (e.g. `Stack`).
    pub struct_type: Path,
    /// The field holding the array base pointer (e.g. `data`).
    pub field: Ident,
    /// The element type (e.g. `i64`).
    pub element_type: Path,
    /// Header struct whose `items` field is element 0, when the base
    /// pointer addresses the header rather than element 0.
    pub header: Option<Path>,
}

/// One entry in `int_fields = { Struct::field => u32, ... }`.
#[derive(Clone)]
pub struct IntFieldEntry {
    /// The struct that owns the field (e.g. `Stack`).
    pub struct_type: Path,
    /// The field name within the struct (e.g. `size`).
    pub field: Ident,
    /// The field's Rust integer type, which supplies its width and sign.
    pub int_type: Ident,
}

impl IntFieldEntry {
    /// `descr.py get_type_flag(FIELDTYPE)` — the signed/unsigned half,
    /// read off the declared Rust type.
    /// `f64` is `lltype.Float`, not an integer field.
    pub(crate) fn is_float(&self) -> bool {
        self.int_type == "f64"
    }

    pub(crate) fn is_signed(&self) -> syn::Result<bool> {
        match self.int_type.to_string().as_str() {
            "i8" | "i16" | "i32" | "i64" | "isize" => Ok(true),
            "u8" | "u16" | "u32" | "u64" | "usize" | "bool" => Ok(false),
            // `lltype.Float` (`descr.py` `get_type_flag` `FLAG_FLOAT`).
            // Signedness does not apply; the lowerer emits `getfield_gc_f`.
            "f64" => Ok(false),
            other => Err(syn::Error::new_spanned(
                &self.int_type,
                format!("int_fields: `{other}` is not an integer or f64 type"),
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallPolicyKind {
    ResidualVoid,
    ResidualVoidWrapped,
    /// `EF_CANNOT_RAISE` (`call.py getcalldescr`'s non-elidable
    /// `else` branch).  Producers pick this when they statically know
    /// the callee cannot raise but is otherwise neither elidable nor
    /// loop-invariant — e.g. flat TLS / buffer shims.  Maps to
    /// [`CondCallEffectSlot::CannotRaise`].
    ResidualVoidCannotRaise,
    ResidualVoidCannotRaiseWrapped,
    MayForceVoid,
    MayForceVoidWrapped,
    LoopInvariantVoid,
    LoopInvariantVoidWrapped,
    ResidualInt,
    ResidualIntWrapped,
    /// `EF_CANNOT_RAISE` (`call.py getcalldescr`'s non-elidable
    /// `else` branch) for int-returning residual helpers.  Mirrors
    /// the void-side `ResidualVoidCannotRaise` pair.  Producers pick
    /// this when the callee is statically known to be non-elidable
    /// AND cannot raise.
    ResidualIntCannotRaise,
    ResidualIntCannotRaiseWrapped,
    MayForceInt,
    MayForceIntWrapped,
    LoopInvariantInt,
    LoopInvariantIntWrapped,
    // `EF_ELIDABLE_CAN_RAISE` (call.py:297). Default elidable variant —
    // emits trailing `GUARD_NO_EXCEPTION`.
    ElidableInt,
    ElidableIntWrapped,
    // `EF_ELIDABLE_CANNOT_RAISE` (call.py:299). Skips the
    // `GUARD_NO_EXCEPTION` because `effectinfo.check_can_raise(False)`
    // is false for `extraeffect == 0`.
    ElidableIntCannotRaise,
    ElidableIntCannotRaiseWrapped,
    // `EF_ELIDABLE_OR_MEMORYERROR` (call.py:295). Same dispatch as
    // `_can_raise` but distinguishes memory-only failure modes.
    ElidableIntOrMemerror,
    ElidableIntOrMemerrorWrapped,
    ResidualRef,
    /// Like `ResidualRef` but stamps the descr EffectInfo with
    /// `RuntimeHelperKind::NurseryAlloc`, which the backend lowers to an inline
    /// nursery bump. Result is a
    /// real escaping Ref — no virtualization.
    NurseryAllocRef,
    /// Fresh allocation that is not a fixed-size nursery bump. Empty write
    /// sets (`can_raise_effect_info`, `analyze_external_call` `bottom_result`)
    /// so the call does not flush cached fields, and the compiled code still
    /// enters the function.
    AllocRef,
    /// `rstr.py ll_strconcat` `@jit.oopspec('stroruni.concat(s1, s2)')`.
    /// Unwrapped ref result; EffectInfo is `EF_ELIDABLE_OR_MEMORYERROR` +
    /// `OS_STR_CONCAT` so `OptString` `opt_call_stroruni_STR_CONCAT` sees it.
    StrConcat,
    ResidualRefWrapped,
    /// `EF_CANNOT_RAISE` for ref-returning residual helpers.
    /// Mirrors `ResidualIntCannotRaiseWrapped`; the unwrapped variant
    /// is absent because the inferred lower path cannot recover a
    /// static ref-return BindingKind from the policy byte alone
    /// (mirrors the existing `ResidualRefWrapped`-only shape).
    ResidualRefCannotRaiseWrapped,
    MayForceRefWrapped,
    LoopInvariantRefWrapped,
    ElidableRefWrapped,
    ElidableRefCannotRaiseWrapped,
    ElidableRefOrMemerrorWrapped,
    ResidualFloatWrapped,
    /// `EF_CANNOT_RAISE` for float-returning residual helpers.
    /// Mirrors `ResidualIntCannotRaiseWrapped` / `ResidualRefCannotRaiseWrapped`.
    ResidualFloatCannotRaiseWrapped,
    MayForceFloatWrapped,
    LoopInvariantFloatWrapped,
    ElidableFloatWrapped,
    ElidableFloatCannotRaiseWrapped,
    ElidableFloatOrMemerrorWrapped,
    InlineInt,
    InlineRef,
    InlineFloat,
    InlineVoid,
    /// `BC_INLINE_CALL` ('j' argcode) into a pipeline-built sub-jitcode
    /// resolved by name through the host's `__majit_pipeline_jitcode`, rather
    /// than a macro-generated `__majit_inline_jitcode_<name>` helper. Int /
    /// Ref / Float select the trailing-return register kind read back into the
    /// caller binding; Void records a statement-position call with no result.
    /// `make_jitcodes()` (codewriter.py) builds the callee; the dispatch
    /// traces into it the same way `Inline*` does, only the jitcode source
    /// differs.
    InlinePipelineInt,
    InlinePipelineRef,
    InlinePipelineFloat,
    InlinePipelineVoid,
    /// Calls the function on the concrete path only; the JIT-path lowerer
    /// emits no IR ops at all. Use for operations that have no JIT-visible
    /// side effects (e.g. returning a node to a free-list -- RPython's `del`
    /// generates zero IR ops).
    ConcreteOnlyVoid,
}

pub(crate) fn parse_call_policy_kind(kind: &Ident) -> Option<CallPolicyKind> {
    Some(match kind.to_string().as_str() {
        "residual_void" => CallPolicyKind::ResidualVoid,
        "residual_void_wrapped" => CallPolicyKind::ResidualVoidWrapped,
        "residual_void_cannot_raise" => CallPolicyKind::ResidualVoidCannotRaise,
        "residual_void_cannot_raise_wrapped" => CallPolicyKind::ResidualVoidCannotRaiseWrapped,
        "may_force_void" => CallPolicyKind::MayForceVoid,
        "may_force_void_wrapped" => CallPolicyKind::MayForceVoidWrapped,
        "loopinvariant_void" => CallPolicyKind::LoopInvariantVoid,
        "loopinvariant_void_wrapped" => CallPolicyKind::LoopInvariantVoidWrapped,
        "residual_int" => CallPolicyKind::ResidualInt,
        "residual_int_wrapped" => CallPolicyKind::ResidualIntWrapped,
        "residual_int_cannot_raise" => CallPolicyKind::ResidualIntCannotRaise,
        "residual_int_cannot_raise_wrapped" => CallPolicyKind::ResidualIntCannotRaiseWrapped,
        "may_force_int" => CallPolicyKind::MayForceInt,
        "may_force_int_wrapped" => CallPolicyKind::MayForceIntWrapped,
        "loopinvariant_int" => CallPolicyKind::LoopInvariantInt,
        "loopinvariant_int_wrapped" => CallPolicyKind::LoopInvariantIntWrapped,
        // `call.py _canraise(op)` 3-way pick on the elidable
        // branch. `elidable_*` (no suffix) is the EF_ELIDABLE_CAN_RAISE
        // default; `_cannot_raise` / `_or_memerror` map to
        // EF_ELIDABLE_CANNOT_RAISE / EF_ELIDABLE_OR_MEMORYERROR.
        "elidable_int" => CallPolicyKind::ElidableInt,
        "elidable_int_wrapped" => CallPolicyKind::ElidableIntWrapped,
        "elidable_int_cannot_raise" => CallPolicyKind::ElidableIntCannotRaise,
        "elidable_int_cannot_raise_wrapped" => CallPolicyKind::ElidableIntCannotRaiseWrapped,
        "elidable_int_or_memerror" => CallPolicyKind::ElidableIntOrMemerror,
        "elidable_int_or_memerror_wrapped" => CallPolicyKind::ElidableIntOrMemerrorWrapped,
        "residual_ref" => CallPolicyKind::ResidualRef,
        "nursery_alloc_ref" => CallPolicyKind::NurseryAllocRef,
        "alloc_ref" => CallPolicyKind::AllocRef,
        "str_concat" => CallPolicyKind::StrConcat,
        "residual_ref_wrapped" => CallPolicyKind::ResidualRefWrapped,
        "residual_ref_cannot_raise_wrapped" => CallPolicyKind::ResidualRefCannotRaiseWrapped,
        "may_force_ref_wrapped" => CallPolicyKind::MayForceRefWrapped,
        "loopinvariant_ref_wrapped" => CallPolicyKind::LoopInvariantRefWrapped,
        "elidable_ref_wrapped" => CallPolicyKind::ElidableRefWrapped,
        "elidable_ref_cannot_raise_wrapped" => CallPolicyKind::ElidableRefCannotRaiseWrapped,
        "elidable_ref_or_memerror_wrapped" => CallPolicyKind::ElidableRefOrMemerrorWrapped,
        "residual_float_wrapped" => CallPolicyKind::ResidualFloatWrapped,
        "residual_float_cannot_raise_wrapped" => CallPolicyKind::ResidualFloatCannotRaiseWrapped,
        "may_force_float_wrapped" => CallPolicyKind::MayForceFloatWrapped,
        "loopinvariant_float_wrapped" => CallPolicyKind::LoopInvariantFloatWrapped,
        "elidable_float_wrapped" => CallPolicyKind::ElidableFloatWrapped,
        "elidable_float_cannot_raise_wrapped" => CallPolicyKind::ElidableFloatCannotRaiseWrapped,
        "elidable_float_or_memerror_wrapped" => CallPolicyKind::ElidableFloatOrMemerrorWrapped,
        "inline_int" => CallPolicyKind::InlineInt,
        "inline_ref" => CallPolicyKind::InlineRef,
        "inline_float" => CallPolicyKind::InlineFloat,
        "inline_void" => CallPolicyKind::InlineVoid,
        "inline_pipeline_int" => CallPolicyKind::InlinePipelineInt,
        "inline_pipeline_ref" => CallPolicyKind::InlinePipelineRef,
        "inline_pipeline_float" => CallPolicyKind::InlinePipelineFloat,
        "inline_pipeline_void" => CallPolicyKind::InlinePipelineVoid,
        "concrete_only_void" => CallPolicyKind::ConcreteOnlyVoid,
        _ => return None,
    })
}

impl Parse for JitInterpConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut trace_cfg = None;
        let mut state_type = None;
        let mut env_type = None;
        let mut io_shims = None;
        let mut calls: Vec<CallEntry> = Vec::new();
        let mut auto_calls = None;
        let mut greens = None;
        let mut reds = None;
        let mut virtualizable_decl = None;
        let mut state_fields = None;
        let mut recover: Option<Path> = None;
        let mut recursive_entry: Option<Path> = None;
        let mut residual_writes: Vec<ResidualWriteEntry> = Vec::new();
        let mut pool_arrays: Vec<PoolArrayEntry> = Vec::new();
        let mut ref_fields: Vec<RefFieldEntry> = Vec::new();
        let mut array_fields: Vec<ArrayFieldEntry> = Vec::new();
        let mut int_fields: Vec<IntFieldEntry> = Vec::new();
        let mut call_returns: Vec<(Path, Path)> = Vec::new();
        let mut struct_allocs: Vec<(Path, Path)> = Vec::new();
        let mut headerless_structs: Vec<Path> = Vec::new();
        let mut inlined_prefix: Vec<InlinedPrefixEntry> = Vec::new();
        let mut native_int_binops: Vec<(Path, Ident)> = Vec::new();
        let mut native_tag_small: Vec<Path> = Vec::new();
        let mut native_identity: Vec<Path> = Vec::new();
        let mut split_dispatch = false;
        let mut switch_dispatch = false;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "state" => {
                    state_type = Some(input.parse::<Ident>()?);
                }
                "env" => {
                    env_type = Some(input.parse::<Ident>()?);
                }
                "io_shims" => {
                    io_shims = Some(parse_io_shim_map(input)?);
                }
                "calls" => {
                    calls.extend(parse_call_map(input)?);
                }
                "helpers" => {
                    calls.extend(parse_helpers_list(input)?);
                }
                "trace_cfg" => {
                    let content;
                    syn::parenthesized!(content in input);
                    trace_cfg = Some(content.parse()?);
                }
                "auto_calls" => {
                    auto_calls = Some(input.parse::<LitBool>()?.value);
                }
                "greens" => {
                    let specs = green_type_tag::parse_green_spec_list(input)?;
                    greens = Some(specs);
                }
                "reds" => {
                    if reds.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `reds`"));
                    }
                    reds = Some(parse_expr_list(input)?);
                }
                "virtualizable_fields" => {
                    virtualizable_decl = Some(parse_virtualizable_decl(input)?);
                }
                "state_fields" => {
                    state_fields = Some(parse_state_fields(input)?);
                }
                "recover" => {
                    recover = Some(input.parse::<Path>()?);
                }
                "recursive_entry" => {
                    recursive_entry = Some(input.parse::<Path>()?);
                }
                "residual_writes" => {
                    residual_writes = parse_residual_writes_map(input)?;
                }
                "pool_arrays" => {
                    pool_arrays = parse_pool_arrays_map(input)?;
                }
                "ref_fields" => {
                    ref_fields = parse_ref_fields_map(input)?;
                }
                "array_fields" => {
                    array_fields = parse_array_fields_map(input)?;
                }
                "int_fields" => {
                    int_fields = parse_int_fields_map(input)?;
                }
                "call_returns" => {
                    call_returns = parse_call_returns_map(input)?;
                }
                "struct_allocs" => {
                    struct_allocs = parse_call_returns_map(input)?;
                }
                "headerless_structs" => {
                    headerless_structs = parse_path_set(input)?;
                }
                "inlined_prefix" => {
                    inlined_prefix = parse_inlined_prefix_map(input)?;
                }
                "native_int_binops" => {
                    native_int_binops = parse_native_int_binops_map(input)?;
                }
                "native_tag_small" => {
                    native_tag_small = parse_native_tag_small_list(input)?;
                }
                "native_identity" => {
                    native_identity = parse_path_set(input)?;
                }
                "split_dispatch" => {
                    split_dispatch = input.parse::<LitBool>()?.value;
                }
                "switch_dispatch" => {
                    switch_dispatch = input.parse::<LitBool>()?.value;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_interp parameter: `{other}`"),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        let state_type =
            state_type.ok_or_else(|| syn::Error::new(input.span(), "missing `state` parameter"))?;
        let env_type =
            env_type.ok_or_else(|| syn::Error::new(input.span(), "missing `env` parameter"))?;

        if state_fields.is_none() {
            return Err(syn::Error::new(
                input.span(),
                "missing `state_fields` parameter",
            ));
        }

        // Read the presence of the key BEFORE the `Option` is collapsed:
        // `unwrap_or_default()` below maps an absent `greens` and an explicit
        // `greens = []` onto the same empty `Vec`, and downstream there is
        // nothing left to recover the difference from.
        let greens_declared = greens.is_some();
        let greens_specs = greens.unwrap_or_default();
        let (green_exprs, green_type_tags): (Vec<Expr>, Vec<Option<green_type_tag::GreenTypeTag>>) =
            greens_specs
                .into_iter()
                .map(|spec| (spec.expr, spec.type_tag))
                .unzip();

        reject_overlapping_call_vocabularies(
            &calls,
            &native_int_binops,
            &native_tag_small,
            &native_identity,
        )?;

        Ok(JitInterpConfig {
            trace_cfg,
            state_type,
            env_type,
            io_shims: io_shims.unwrap_or_default(),
            calls,
            auto_calls: auto_calls.unwrap_or(false),
            greens: green_exprs,
            greens_declared,
            reds: reds.unwrap_or_default(),
            green_type_tags,
            virtualizable_decl,
            state_fields,
            recover,
            recursive_entry,
            residual_writes,
            pool_arrays,
            ref_fields,
            array_fields,
            int_fields,
            call_returns,
            struct_allocs,
            headerless_structs,
            inlined_prefix,
            native_int_binops,
            native_tag_small,
            native_identity,
            split_dispatch,
            switch_dispatch,
        })
    }
}

/// Parse `residual_writes = { <ref_scalar>.<field> [@ StructType] => [helper, ...], ... }`.
/// Each map key is a `state` ref-scalar field path (`selected_ref.size`) and
/// the value is a bracketed list of residual helper function paths that mutate
/// it.  See [`ResidualWriteEntry`].
fn parse_residual_writes_map(input: ParseStream) -> syn::Result<Vec<ResidualWriteEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let ref_scalar: Ident = content.parse()?;
        content.parse::<Token![.]>()?;
        let field: Ident = content.parse()?;
        // `<field>[]` — the elements behind the pointer, not the pointer field.
        let writes_elements = content.peek(syn::token::Bracket);
        if writes_elements {
            let brackets;
            bracketed!(brackets in content);
            if !brackets.is_empty() {
                return Err(brackets.error(
                    "residual_writes element syntax takes no index: write `field[]`, \
                     which declares a write to some element of the array",
                ));
            }
        }
        let struct_type = if content.peek(Token![@]) {
            content.parse::<Token![@]>()?;
            Some(content.parse::<Path>()?)
        } else {
            None
        };
        content.parse::<Token![=>]>()?;
        let helpers_content;
        bracketed!(helpers_content in content);
        let helpers: Punctuated<Path, Token![,]> =
            helpers_content.parse_terminated(Path::parse, Token![,])?;
        entries.push(ResidualWriteEntry {
            ref_scalar,
            field,
            struct_type,
            writes_elements,
            helpers: helpers.into_iter().collect(),
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

/// Parse `pool_arrays = { <ref>.<items>[<len>] => <getter> [-> ElementType],
/// ... }`.  Each entry maps a `ref(T)` state scalar (the array base) and the
/// array field inside it to the marker function whose call indexes it,
/// optionally naming a length word ahead of the items and tagging the loaded
/// element's struct type.
///
/// `<items>` is mandatory: the read has to add some offset to the base, and the
/// only alternatives to declaring it are assuming one or hard-coding one.
fn parse_pool_arrays_map(input: ParseStream) -> syn::Result<Vec<PoolArrayEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let base: Ident = content.parse()?;
        content.parse::<Token![.]>()?;
        let items_field: Ident = content.parse()?;
        let len_field = if content.peek(syn::token::Bracket) {
            let len;
            syn::bracketed!(len in content);
            Some(len.parse::<Ident>()?)
        } else {
            None
        };
        content.parse::<Token![=>]>()?;
        let getter: Path = content.parse()?;
        let element_type = if content.peek(Token![->]) {
            content.parse::<Token![->]>()?;
            Some(content.parse::<Path>()?)
        } else {
            None
        };
        entries.push(PoolArrayEntry {
            base,
            getter,
            items_field,
            len_field,
            element_type,
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

/// Parse `ref_fields = { Struct::field => PointeeType, ... }`.
/// Each entry declares that `Struct.field` is a ref-kind (pointer) field
/// whose pointee is `PointeeType`.  The path `Struct::field` is parsed as
/// a single `Path` and then split: the last segment is the field name, the
/// remaining prefix is the struct type.
/// Parse `call_returns = { func_path => StructType, ... }`.
pub(crate) fn parse_call_returns_map(input: ParseStream) -> syn::Result<Vec<(Path, Path)>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let func_path: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let return_type: Path = content.parse()?;
        entries.push((func_path, return_type));
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_native_int_binops_map(input: ParseStream) -> syn::Result<Vec<(Path, Ident)>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let func_path: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let opcode: Ident = content.parse()?;
        entries.push((func_path, opcode));
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_native_tag_small_list(input: ParseStream) -> syn::Result<Vec<Path>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let func_path: Path = content.parse()?;
        entries.push(func_path);
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_path_set(input: ParseStream) -> syn::Result<Vec<Path>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        entries.push(content.parse::<Path>()?);
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

fn split_struct_field_path(full_path: Path) -> syn::Result<(Path, Ident)> {
    let mut segments: Vec<_> = full_path.segments.into_iter().collect();
    if segments.len() < 2 {
        return Err(syn::Error::new_spanned(
            full_path.leading_colon,
            "entry must be `Struct::field`",
        ));
    }
    let field_seg = segments.pop().unwrap();
    let field = field_seg.ident;
    let struct_type = syn::Path {
        leading_colon: full_path.leading_colon,
        segments: segments.into_iter().collect(),
    };
    Ok((struct_type, field))
}

/// Parse `int_fields = { Struct::field => u32, ... }`.  Split like
/// `ref_fields`: the last segment of `Struct::field` is the field name.
pub(crate) fn parse_int_fields_map(input: ParseStream) -> syn::Result<Vec<IntFieldEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let (struct_type, field) = split_struct_field_path(content.parse::<Path>()?)?;
        content.parse::<Token![=>]>()?;
        let int_type: Ident = content.parse()?;
        let entry = IntFieldEntry {
            struct_type,
            field,
            int_type,
        };
        entry.is_signed()?;
        entries.push(entry);
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

/// The struct that DECLARES `field`, following inlined leading substructures
/// outward-in.
///
/// `rclass.py InstanceRepr.getfield` resolves a field against the repr
/// that owns it: `if attr in self.fields` … else recurse to `self.rbase` with
/// `force_cast=True`.  That cast is why `jtransform.py:881` always reads the
/// descr off the declaring struct, and so why one physical field has one
/// descriptor however many structs embed it.
///
/// Shared by the JIT lowerer and by both concrete-path rewriters, which walk
/// the same source and must agree about where a field lives.  They disagree
/// only where one stopped early, and a disagreement here is not a build error
/// — it is a dispatch arm that silently degrades.
///
/// `declares` answers whether a struct declares the field itself; each caller
/// supplies it over whatever declaration maps it holds.  Returns `struct_path`
/// unchanged when it declares the field, when it embeds no declared base, or
/// when nothing declared one.
pub(crate) fn declaring_struct_of(
    inlined_prefix: &std::collections::HashMap<Vec<String>, (String, Path)>,
    declares: impl Fn(&Path, &str) -> bool,
    struct_path: &Path,
    field: &str,
) -> Path {
    let mut current = struct_path.clone();
    // A chain visits a struct at most once before it would repeat, so it cannot
    // be longer than the number of declarations plus the struct it started at.
    // The bound is what makes this terminate: a well-formed chain ends because
    // the Rust types would otherwise be infinitely sized, but the map it walks
    // is written by hand and `A::base => A` names a cycle the Rust types never
    // see. Without the bound that spins the proc macro forever with no output.
    for _ in 0..=inlined_prefix.len() {
        if declares(&current, field) {
            return current;
        }
        let segments: Vec<String> = current
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        let Some((_, base)) = inlined_prefix_entry(inlined_prefix, &segments) else {
            return current;
        };
        current = base.clone();
    }
    current
}

/// Resolve an inlined-base declaration across equivalent Rust path spellings.
///
/// A function body can name an imported type as `Queue` while the macro
/// configuration names it as `crate::storage::Queue`. An exact match remains
/// authoritative. A leaf-name fallback is accepted only when this macro's
/// declaration set contains one type with that leaf; two same-named types make
/// the fallback ambiguous and therefore unresolved.
pub(crate) fn inlined_prefix_entry<'a>(
    inlined_prefix: &'a std::collections::HashMap<Vec<String>, (String, Path)>,
    segments: &[String],
) -> Option<&'a (String, Path)> {
    if let Some(entry) = inlined_prefix.get(segments) {
        return Some(entry);
    }
    let leaf = segments.last()?;
    let mut matches = inlined_prefix
        .iter()
        .filter(|(candidate, _)| candidate.last() == Some(leaf))
        .map(|(_, entry)| entry);
    let entry = matches.next()?;
    matches.next().is_none().then_some(entry)
}

/// Every `"StructLast::field"` key the three field vocabularies declare.
///
/// What a declaring-struct walk needs is only whether a struct declares a
/// field, not what it declared it as, so the three maps collapse to one key
/// set for that purpose.
pub(crate) fn declared_field_keys(
    ref_fields: &[RefFieldEntry],
    int_fields: &[IntFieldEntry],
    array_fields: &[ArrayFieldEntry],
) -> std::collections::HashSet<String> {
    fn key(struct_type: &Path, field: &Ident) -> String {
        let struct_last = struct_type
            .segments
            .last()
            .map(|segment| segment.ident.to_string())
            .unwrap_or_default();
        format!("{struct_last}::{field}")
    }
    ref_fields
        .iter()
        .map(|entry| key(&entry.struct_type, &entry.field))
        .chain(
            int_fields
                .iter()
                .map(|entry| key(&entry.struct_type, &entry.field)),
        )
        .chain(
            array_fields
                .iter()
                .map(|entry| key(&entry.struct_type, &entry.field)),
        )
        .collect()
}

/// Index `inlined_prefix` by the embedding struct's full canonical segments.
///
/// Keyed by every segment rather than by the last one the field vocabularies
/// use: this declaration names a type relationship, and two same-named types in
/// different modules do not share a base.
pub(crate) fn inlined_prefix_index(
    entries: &[InlinedPrefixEntry],
) -> std::collections::HashMap<Vec<String>, (String, Path)> {
    entries
        .iter()
        .map(|entry| {
            (
                entry
                    .outer
                    .segments
                    .iter()
                    .map(|segment| segment.ident.to_string())
                    .collect(),
                (entry.base_field.to_string(), entry.base.clone()),
            )
        })
        .collect()
}

/// Parse `inlined_prefix = { Outer::field => Base, ... }`.  Split like
/// `ref_fields`: the last segment of `Outer::field` names the field holding the
/// inlined base.
pub(crate) fn parse_inlined_prefix_map(input: ParseStream) -> syn::Result<Vec<InlinedPrefixEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let (outer, base_field) = split_struct_field_path(content.parse::<Path>()?)?;
        content.parse::<Token![=>]>()?;
        let base: Path = content.parse()?;
        entries.push(InlinedPrefixEntry {
            outer,
            base_field,
            base,
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_ref_fields_map(input: ParseStream) -> syn::Result<Vec<RefFieldEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let (struct_type, field) = split_struct_field_path(content.parse::<Path>()?)?;
        content.parse::<Token![=>]>()?;
        let pointee_type: Path = content.parse()?;
        entries.push(RefFieldEntry {
            struct_type,
            field,
            pointee_type,
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

/// Parse `array_fields = { Struct::field => ElementType, ... }`.
/// Each entry declares that `Struct.field` is the base pointer of a
/// contiguous `ElementType` array; see [`ArrayFieldEntry`].
/// `ElementType in Header` names the struct whose `items` field is element 0.
pub(crate) fn parse_array_fields_map(input: ParseStream) -> syn::Result<Vec<ArrayFieldEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let (struct_type, field) = split_struct_field_path(content.parse::<Path>()?)?;
        content.parse::<Token![=>]>()?;
        let element_type: Path = content.parse()?;
        let header = if content.peek(Token![in]) {
            content.parse::<Token![in]>()?;
            Some(content.parse()?)
        } else {
            None
        };
        entries.push(ArrayFieldEntry {
            struct_type,
            field,
            element_type,
            header,
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

fn parse_expr_list(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let content;
    bracketed!(content in input);
    let exprs: Punctuated<Expr, Token![,]> = content.parse_terminated(Expr::parse, Token![,])?;
    Ok(exprs.into_iter().collect())
}

/// Parse virtualizable_fields = { var: IDENT, token_offset: PATH, fields: { ... }, arrays: { ... } }
///
/// Parse `state_fields = { name: type, ... }` where type is `int`, `float`,
/// `[int]`, `[int; virt]`, `[float; virt]`, or `opaque(TypePath)`.
fn parse_state_fields(input: ParseStream) -> syn::Result<StateFieldsConfig> {
    let content;
    braced!(content in input);
    let mut fields = Vec::new();

    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        let kind = if content.peek(syn::token::Bracket) {
            // Array: [int]/[float] or virtualizable: [int; virt]/[float; virt]
            let inner;
            bracketed!(inner in content);
            let item_type: Ident = inner.parse()?;
            if item_type != "int" && item_type != "float" {
                return Err(syn::Error::new(
                    item_type.span(),
                    format!(
                        "state_fields array `{name}` uses unsupported item type `{item_type}`; \
                         supported item types: `int`, `float`"
                    ),
                ));
            }
            if inner.peek(Token![;]) {
                inner.parse::<Token![;]>()?;
                let flag: Ident = inner.parse()?;
                if flag == "virt" {
                    StateFieldKind::VirtArray(item_type)
                } else {
                    return Err(syn::Error::new(
                        flag.span(),
                        format!("unknown array modifier `{flag}`, expected `virt`"),
                    ));
                }
            } else {
                StateFieldKind::Array(item_type)
            }
        } else if content.peek(Token![ref]) {
            // `ref(<TypePath>)` — a ref-typed scalar carried in the ref
            // register bank.  `ref` is a reserved keyword, so it is matched
            // with `Token![ref]` before the identifier-based scalar forms.
            // Storage on the state struct is a usize carrier; `T` names the
            // pointed-at heap class.
            content.parse::<Token![ref]>()?;
            let inner;
            syn::parenthesized!(inner in content);
            let type_path: syn::Path = inner.parse()?;
            StateFieldKind::Ref(type_path)
        } else {
            // Scalar forms: `int`, `int(<TypePath>)`, `float`,
            // `float(<TypePath>)`, or `opaque(TypePath)`.
            // `str` is a keyword, so `Ident::parse` rejects `s: str`.
            let head = syn::Ident::parse_any(&content)?;
            if head == "str" {
                StateFieldKind::Str
            } else if head == "opaque" {
                let inner;
                syn::parenthesized!(inner in content);
                let type_path: syn::Path = inner.parse()?;
                StateFieldKind::Opaque(type_path)
            } else if head == "int" || head == "float" {
                // RPython parity: optional `int(<TypePath>)` declares the
                // Rust storage type when it differs from i64. `float(<Path>)`
                // mirrors that shape; bare `float` stores as f64.
                let rust_type = if content.peek(syn::token::Paren) {
                    let inner;
                    syn::parenthesized!(inner in content);
                    Some(inner.parse::<syn::Path>()?)
                } else {
                    None
                };
                StateFieldKind::Scalar {
                    ir_type: head,
                    rust_type,
                }
            } else {
                return Err(syn::Error::new(
                    head.span(),
                    format!(
                        "state_fields scalar `{name}` uses unsupported type `{head}`; \
                         supported: `int`, `int(<TypePath>)`, `float`, \
                         `float(<TypePath>)`, `opaque(TypePath)`, `str`"
                    ),
                ));
            }
        };

        fields.push(StateFieldDecl { name, kind });
        let _ = content.parse::<Token![,]>();
    }

    Ok(StateFieldsConfig { fields })
}

/// RPython equivalent: VirtualizableInfo construction from virtualizable.py
/// + jtransform.py's field-to-descriptor mapping.
fn parse_virtualizable_decl(input: ParseStream) -> syn::Result<VirtualizableDecl> {
    let content;
    braced!(content in input);

    let mut var_name = None;
    let mut token_offset = None;
    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    while !content.is_empty() {
        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "var" => {
                var_name = Some(content.parse::<Ident>()?);
            }
            "token_offset" => {
                token_offset = Some(content.parse::<Path>()?);
            }
            "fields" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let field_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    fields.push(VableFieldDecl {
                        name,
                        field_type,
                        offset,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            "arrays" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let item_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let field_offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    let layout = if inner.peek(syn::token::Brace) {
                        let layout_content;
                        braced!(layout_content in inner);
                        let mut ptr_offset = None;
                        let mut length_offset = None;
                        let mut items_offset = None;
                        while !layout_content.is_empty() {
                            let layout_key: Ident = layout_content.parse()?;
                            layout_content.parse::<Token![:]>()?;
                            match layout_key.to_string().as_str() {
                                "ptr_offset" => {
                                    ptr_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "length_offset" => {
                                    length_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "items_offset" => {
                                    items_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                other => {
                                    return Err(syn::Error::new(
                                        layout_key.span(),
                                        format!(
                                            "unknown virtualizable array layout parameter: `{other}`"
                                        ),
                                    ));
                                }
                            }
                            let _ = layout_content.parse::<Token![,]>();
                        }
                        if let Some(ptr_offset) = ptr_offset {
                            VableArrayLayoutDecl::Embedded {
                                field_offset,
                                ptr_offset,
                                length_offset: length_offset.ok_or_else(|| {
                                    syn::Error::new(
                                        inner.span(),
                                        "missing `length_offset` in embedded virtualizable array layout",
                                    )
                                })?,
                                items_offset: items_offset.ok_or_else(|| {
                                    syn::Error::new(
                                        inner.span(),
                                        "missing `items_offset` in embedded virtualizable array layout",
                                    )
                                })?,
                            }
                        } else {
                            VableArrayLayoutDecl::Direct {
                                field_offset,
                                length_offset,
                                items_offset,
                            }
                        }
                    } else {
                        VableArrayLayoutDecl::Direct {
                            field_offset,
                            length_offset: None,
                            items_offset: None,
                        }
                    };
                    arrays.push(VableArrayDecl {
                        name,
                        item_type,
                        layout,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown virtualizable_fields parameter: `{other}`"),
                ));
            }
        }
        let _ = content.parse::<Token![,]>();
    }

    let var_name = var_name
        .ok_or_else(|| syn::Error::new(content.span(), "missing `var` in virtualizable_fields"))?;
    let token_offset = token_offset.ok_or_else(|| {
        syn::Error::new(
            content.span(),
            "missing `token_offset` in virtualizable_fields",
        )
    })?;

    Ok(VirtualizableDecl {
        var_name,
        token_offset,
        fields,
        arrays,
    })
}

/// Parse `{ path::func => jit_func, ... }`.
fn parse_io_shim_map(input: ParseStream) -> syn::Result<Vec<(Path, Ident)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let shim: Ident = content.parse()?;
        map.push((func, shim));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `{ path::func, path::func => residual_int, ... }`.
///
/// Per-entry forms:
///   - `path::func`                (default policy)
///   - `path::func => policy_kind` (explicit policy)
fn parse_call_map(input: ParseStream) -> syn::Result<Vec<CallEntry>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        let policy = if content.peek(Token![=>]) {
            content.parse::<Token![=>]>()?;
            let kind: Ident = content.parse()?;
            Some(parse_call_policy_kind(&kind).ok_or_else(|| {
                syn::Error::new(
                    kind.span(),
                    "call policy must be a supported residual/may_force/release_gil/loopinvariant policy or inline_int/inline_ref/inline_float",
                )
            })?)
        } else {
            None
        };
        map.push(CallEntry { path: func, policy });
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `[func_a, func_b, func_c]` — shorthand helper list with auto-inferred policies.
fn parse_helpers_list(input: ParseStream) -> syn::Result<Vec<CallEntry>> {
    let content;
    bracketed!(content in input);
    let paths: Punctuated<Path, Token![,]> = content.parse_terminated(Path::parse, Token![,])?;
    Ok(paths
        .into_iter()
        .map(|p| CallEntry {
            path: p,
            policy: None,
        })
        .collect())
}

/// Reject a plain `[int]` state-array that is stored-to inside the traced
/// loop. Such an array is *loop-carried*, and the observable — measured on
/// `majit/examples/tinyframe` with this refusal disabled — is that the element
/// does not read back its heap value: three committed value tests answer 0
/// where the program answers 40, 40 and 5, and they answer 0 whatever the
/// array held, including when the register is seeded through the mainloop's
/// own input ahead of every candidate arming pc. The supported declaration for
/// a mutated, loop-carried array is `[int; virt]`, which answers correctly on
/// the same fixtures. A plain `[int]` array that is only *read* in the loop
/// (or only written before it) is loop-invariant and stays valid, so the check
/// fires solely on stores reached by the dispatch loop the lowerer traces.
///
/// Do not read the refusal as asking for a restore path. In the measured
/// subject the trace carries no array operation at all — the recorded body is
/// `IntAdd, IntGt, GuardTrue, GuardFutureCondition, Jump`, the register file
/// is traced as loop-carried inputargs, and nothing lowers to a state-array
/// load or store — so there is no array write for a deopt to fail to restore.
/// What the 0 actually is remains UNMEASURED: the `[int; virt]` arm was never
/// logged beside the plain one, so the two op inventories and the two guards'
/// resume payloads have never been compared.
///
/// `validate_rejects_loop_carried_plain_array`, `validate_accepts_virt_array`
/// and `validate_accepts_read_only_plain_array` grade *which declarations this
/// function refuses*; none of them grades whether the refused declaration
/// miscompiles. This is a selector with no classifier, and the readback above
/// is the first witness it has ever had.
fn validate_state_fields(config: &JitInterpConfig, func: &ItemFn) -> syn::Result<()> {
    let Some(sf) = &config.state_fields else {
        return Ok(());
    };
    let plain_arrays: std::collections::HashSet<String> = sf
        .fields
        .iter()
        .filter(|f| matches!(f.kind, StateFieldKind::Array(_)))
        .map(|f| f.name.to_string())
        .collect();
    if plain_arrays.is_empty() {
        return Ok(());
    }
    // Only stores reached by the traced dispatch loop are a hazard; init
    // stores before the loop are not traced.
    let Some(loop_body) = find_traced_loop_body(&func.block) else {
        return Ok(());
    };
    let mut finder = PlainArrayStoreFinder {
        plain_arrays: &plain_arrays,
        offender: None,
    };
    syn::visit::Visit::visit_block(&mut finder, loop_body);
    if let Some((name, span)) = finder.offender {
        return Err(syn::Error::new(
            span,
            format!(
                "state field `{name}` is a plain `[int]` array stored inside the traced loop, \
                 so it is loop-carried. A loop-carried plain `[int]` element does not read back \
                 its heap value: the traced loop answers as if the element were 0, whatever the \
                 array holds, silently miscompiling. Declare it as `[int; virt]`, which answers \
                 correctly on the same program."
            ),
        ));
    }
    Ok(())
}

/// The `loop {}` / `while {}` body that contains the `jit_merge_point!()`
/// marker — the region the lowerer traces.
fn find_traced_loop_body(block: &syn::Block) -> Option<&syn::Block> {
    for stmt in &block.stmts {
        let syn::Stmt::Expr(expr, _) = stmt else {
            continue;
        };
        let body = match expr {
            Expr::Loop(l) => &l.body,
            Expr::While(w) => &w.body,
            _ => continue,
        };
        let has_merge_point = body
            .stmts
            .iter()
            .any(|s| matches!(s, syn::Stmt::Macro(m) if m.mac.path.is_ident("jit_merge_point")));
        if has_merge_point {
            return Some(body);
        }
    }
    None
}

struct PlainArrayStoreFinder<'a> {
    plain_arrays: &'a std::collections::HashSet<String>,
    offender: Option<(String, proc_macro2::Span)>,
}

impl<'a> PlainArrayStoreFinder<'a> {
    /// `state.<field>[_]` with `<field>` a plain array → the field name.
    fn plain_array_index_target(&self, left: &Expr) -> Option<String> {
        let Expr::Index(idx) = left else {
            return None;
        };
        let Expr::Field(f) = &*idx.expr else {
            return None;
        };
        if !matches!(&*f.base, Expr::Path(p) if p.qself.is_none() && p.path.is_ident("state")) {
            return None;
        }
        let syn::Member::Named(member) = &f.member else {
            return None;
        };
        let name = member.to_string();
        self.plain_arrays.contains(&name).then_some(name)
    }

    fn check_assign_target(&mut self, left: &Expr) {
        if self.offender.is_some() {
            return;
        }
        if let Some(name) = self.plain_array_index_target(left) {
            use syn::spanned::Spanned as _;
            self.offender = Some((name, left.span()));
        }
    }
}

impl<'ast, 'a> syn::visit::Visit<'ast> for PlainArrayStoreFinder<'a> {
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        self.check_assign_target(&node.left);
        syn::visit::visit_expr_assign(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        // Compound assignment (`state.regs[i] += 1`) is a store too.
        use syn::BinOp::*;
        if matches!(
            node.op,
            AddAssign(_)
                | SubAssign(_)
                | MulAssign(_)
                | DivAssign(_)
                | RemAssign(_)
                | BitXorAssign(_)
                | BitAndAssign(_)
                | BitOrAssign(_)
                | ShlAssign(_)
                | ShrAssign(_)
        ) {
            self.check_assign_target(&node.left);
        }
        syn::visit::visit_expr_binary(self, node);
    }
}

/// Main entry point: transform the function with JIT support.
pub fn transform_jit_interp(config: JitInterpConfig, func: ItemFn) -> TokenStream {
    if let Err(err) = validate_state_fields(&config, &func) {
        return err.to_compile_error();
    }
    if let Err(err) = validate_merge_point_forms(&func) {
        return err.to_compile_error();
    }
    let trace_fn = codegen_trace::generate_trace_fn(&config, &func);
    let state_impl = codegen_state::generate_jit_state(&config, &func);
    let merge_wrapper = generate_merge_wrapper(&config, &func);
    let green_key_fn = generate_green_key_fn(&config, &func);
    let transformed_fn = transform_function(&config, &func, true);

    let generated = quote! {
        #state_impl
        #trace_fn
        #merge_wrapper
        #green_key_fn
        #transformed_fn
    };
    if let Some(condition) = &config.trace_cfg {
        let concrete = transform_function(&config, &func, false);
        let generated = crate::gate_generated_items(generated, condition);
        quote! {
            #generated
            #[cfg(not(#condition))]
            #concrete
        }
    } else {
        generated
    }
}

/// Emit `__majit_green_key_<fn>`, which builds the key a merge point of this
/// interpreter files under, for a caller OUTSIDE the mainloop.
///
/// A door that wants to ask whether some position is already compiled has to
/// name that position the way the merge point names it. Without this builder
/// the door has to write the layout out by hand: seed, one `green_uhash_step`
/// per declared green, in declaration order, with the target substituted for a
/// `pc` green. `majit_ir::pypyjit_greenkey` is the same function hard-coded for
/// one consumer's three greens, so a consumer with a different green list cannot
/// reach it; its own documentation says why open-coding the layout is a hazard
/// — "a swapped slot still hashes to something, just not to the same cell".
///
/// The failure is silent and total: the door files under a key nothing else
/// can name, so its probe answers no forever and the artifact it was guarding
/// is never entered. Nothing counts it, because a tier that never enters
/// compiled code still returns the right answers.
///
/// Emitted only where it can be spelled without guessing:
///
/// * Every green must be a plain identifier. The helper's parameters ARE those
///   identifiers, so an expression green has no name to take, and passing the
///   expression's value positionally would reintroduce the ordering hazard this
///   exists to remove.
/// * No green may carry a type tag. A tagged green lowers through an explicit
///   cast (`(#expr) as i64`), which does not compile against the generic
///   parameter the untagged path takes.
///
/// Where either does not hold, no helper appears and a consumer keeps whatever
/// it does today — this adds a way to be right, it does not take one away.
///
/// The greens are the attribute's, so this describes the DEFAULT layout. A
/// merge point that overrides `greens` in its own marker args is not described
/// by it.
fn generate_green_key_fn(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    if !config.greens_declared || config.greens.is_empty() {
        return quote! {};
    }
    if config.green_type_tags.iter().any(Option::is_some) {
        return quote! {};
    }
    let ident_of = |expr: &Expr| -> Option<Ident> {
        match expr {
            Expr::Path(path)
                if path.qself.is_none()
                    && path.attrs.is_empty()
                    && path.path.segments.len() == 1
                    && path.path.segments[0].arguments.is_none() =>
            {
                Some(path.path.segments[0].ident.clone())
            }
            _ => None,
        }
    };
    let Some(greens) = config
        .greens
        .iter()
        .map(&ident_of)
        .collect::<Option<Vec<Ident>>>()
    else {
        return quote! {};
    };

    // Declared greens in declaration order — the same list `green_key_expr`
    // folds at the merge point (`pyjitpl.py` `original_boxes[:num_green_args]`).
    // A driver that needs the loop-header pc declares it (`interp_jit.py`
    // `next_instr`); it is not an extra prepended slot.
    let slots: Vec<Ident> = greens;

    // One parameter per DISTINCT slot, in first-appearance order. A green that
    // repeats is read twice from one argument rather than asked for twice,
    // which is what `Copy` is for.
    let mut params: Vec<Ident> = Vec::new();
    for slot in &slots {
        if !params.contains(slot) {
            params.push(slot.clone());
        }
    }

    // A green that names one of the mainloop's own parameters is typed as that
    // parameter, not generically. It matters most for the case it was written
    // for: a `Ref` green keys on POINTER IDENTITY, so a door holding the same
    // bytes under a different reference type — `&Vec<u8>` where the mainloop
    // took `&[u8]` — hashes to a different cell and probes forever for a key
    // nothing files under. A concrete parameter makes that a type error at the
    // door instead. Greens that name a local (`pc`, usually) have no declared
    // type to take and stay generic.
    let declared_ty = |name: &Ident| -> Option<syn::Type> {
        func.sig.inputs.iter().find_map(|arg| match arg {
            syn::FnArg::Typed(pat) => match &*pat.pat {
                syn::Pat::Ident(id) if id.ident == *name => Some((*pat.ty).clone()),
                _ => None,
            },
            syn::FnArg::Receiver(_) => None,
        })
    };
    let param_decls: Vec<TokenStream> = params
        .iter()
        .map(|name| match declared_ty(name) {
            Some(ty) => quote! { #name: #ty },
            None => quote! { #name: impl majit_ir::GreenAsI64 + ::core::marker::Copy },
        })
        .collect();

    let fn_name = quote::format_ident!("__majit_green_key_{}", func.sig.ident);
    let n = slots.len();
    quote! {
        #[doc(hidden)]
        #[allow(non_snake_case)]
        pub fn #fn_name(
            #(#param_decls),*
        ) -> (u64, [(i64, majit_ir::GreenType); #n]) {
            let __slots = [
                #(<_ as majit_ir::GreenAsI64>::__green_repr(#slots)),*
            ];
            let mut __hash: u64 = majit_ir::GREEN_UHASH_SEED;
            let mut __i = 0usize;
            while __i < #n {
                __hash = majit_ir::green_uhash_step(__hash, __slots[__i].1, __slots[__i].0);
                __i += 1;
            }
            (__hash, __slots)
        }
    }
}

/// Generate a `#[cold]` out-of-line wrapper for the merge_point call.
///
/// This keeps the mainloop hot path thin — only an `is_tracing()` flag check
/// appears inline, while the closure capture and tracing logic live here.
/// Green variables that are portal function parameters rather than the
/// conventional `pc` / `program` pair.  RPython obtains their lltypes from
/// the portal graph signature.  The Rust macro needs an explicit tag for the
/// same reason: it must choose an Int/Ref/Float register bank before rustc
/// type-checks the generated body.
pub(super) fn portal_green_params(
    config: &JitInterpConfig,
    func: &ItemFn,
) -> Vec<(Ident, syn::Type, green_type_tag::GreenTypeTag)> {
    use std::collections::HashMap;

    let param_types: HashMap<String, syn::Type> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            let syn::FnArg::Typed(arg) = arg else {
                return None;
            };
            let syn::Pat::Ident(pat) = &*arg.pat else {
                return None;
            };
            Some((pat.ident.to_string(), (*arg.ty).clone()))
        })
        .collect();

    config
        .greens
        .iter()
        .zip(config.green_type_tags.iter())
        .filter_map(|(expr, tag)| {
            let Expr::Path(path) = expr else {
                return None;
            };
            let ident = path.path.get_ident()?.clone();
            let name = ident.to_string();
            if is_dispatch_entry_green(&name) {
                return None;
            }
            let ty = param_types.get(&name)?.clone();
            let tag = tag.unwrap_or_else(|| {
                panic!(
                    "#[jit_interp] portal-parameter green `{name}` needs an explicit \
                     `: int`, `: ref`, or `: float` tag so the generated JitCode can \
                     assign its register bank (RPython derives this from the portal \
                     graph's lltype)"
                )
            });
            Some((ident, ty, tag))
        })
        .collect()
}

/// Dispatch-entry greens the JitCode already binds as r0 (`program`) and
/// i0 (`pc`). `portal_green_params` uses this filter; loop-carried greens
/// reuse it rather than a second string compare. `warmspot.py
/// portalfunc_ARGS` is positional; this is the existing name-based
/// stand-in for that entry convention (review section 3).
fn is_dispatch_entry_green(name: &str) -> bool {
    name == "pc" || name == "program"
}

struct ScopeBind {
    name: String,
    is_mut: bool,
    ty: Option<syn::Type>,
    init: Option<Expr>,
}

fn expr_single_ident(expr: &Expr) -> Option<String> {
    let Expr::Path(path) = expr else {
        return None;
    };
    path.path.get_ident().map(|ident| ident.to_string())
}

fn local_pat_ident(pat: &syn::Pat) -> Option<(&syn::Ident, bool, Option<&syn::Type>)> {
    match pat {
        syn::Pat::Ident(ident) => Some((&ident.ident, ident.mutability.is_some(), None)),
        syn::Pat::Type(pat_ty) => {
            let syn::Pat::Ident(ident) = &*pat_ty.pat else {
                return None;
            };
            Some((&ident.ident, ident.mutability.is_some(), Some(&pat_ty.ty)))
        }
        _ => None,
    }
}

fn value_kind_from_tag(tag: green_type_tag::GreenTypeTag) -> jitcode_lower::ValueKind {
    use green_type_tag::GreenTypeTag;
    use jitcode_lower::ValueKind;
    match tag {
        GreenTypeTag::Int => ValueKind::Int,
        GreenTypeTag::Ref | GreenTypeTag::Str | GreenTypeTag::Unicode => ValueKind::Ref,
        GreenTypeTag::Float => ValueKind::Float,
    }
}

fn value_kind_from_type(ty: &syn::Type) -> jitcode_lower::ValueKind {
    use jitcode_lower::ValueKind;
    match ty {
        syn::Type::Reference(_) => ValueKind::Ref,
        syn::Type::Path(path) => {
            let name = path
                .path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .unwrap_or_default();
            match name.as_str() {
                "f32" | "f64" => ValueKind::Float,
                _ => ValueKind::Int,
            }
        }
        _ => ValueKind::Int,
    }
}

fn value_kind_from_expr(expr: &Expr) -> Option<jitcode_lower::ValueKind> {
    use jitcode_lower::ValueKind;
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Float(_) => Some(ValueKind::Float),
            syn::Lit::Int(_) | syn::Lit::Bool(_) => Some(ValueKind::Int),
            _ => None,
        },
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            value_kind_from_expr(&unary.expr)
        }
        Expr::Reference(_) => Some(ValueKind::Ref),
        Expr::Cast(cast) => Some(value_kind_from_type(&cast.ty)),
        _ => None,
    }
}

fn last_direct_local_before_portal<'a>(func: &'a ItemFn, name: &str) -> Option<&'a syn::Local> {
    let portal_index = codegen_trace::find_portal_loop(&func.block)?.index;
    let mut found = None;
    for (index, stmt) in func.block.stmts.iter().enumerate() {
        if index == portal_index {
            break;
        }
        let syn::Stmt::Local(local) = stmt else {
            continue;
        };
        if local_pat_ident(&local.pat).is_some_and(|(ident, _, _)| ident == name) {
            found = Some(local);
        }
    }
    found
}

fn classify_green_kind(
    expr: &Expr,
    tag: Option<green_type_tag::GreenTypeTag>,
    func: &ItemFn,
) -> jitcode_lower::ValueKind {
    use jitcode_lower::ValueKind;
    if let Some(tag) = tag {
        return value_kind_from_tag(tag);
    }
    let Some(name) = expr_single_ident(expr) else {
        return value_kind_from_expr(expr).unwrap_or(ValueKind::Int);
    };
    if let Some(ty) = fn_param_type(func, &name) {
        return value_kind_from_type(&ty);
    }
    if let Some(local) = last_direct_local_before_portal(func, &name) {
        if let Some((_, _, ty)) = local_pat_ident(&local.pat)
            && let Some(ty) = ty
        {
            return value_kind_from_type(ty);
        }
        if let Some(init) = local.init.as_ref()
            && let Some(kind) = value_kind_from_expr(&init.expr)
        {
            return kind;
        }
    }
    ValueKind::Int
}

fn fn_param_type(func: &ItemFn, name: &str) -> Option<syn::Type> {
    func.sig.inputs.iter().find_map(|arg| {
        let syn::FnArg::Typed(arg) = arg else {
            return None;
        };
        let (ident, _, _) = local_pat_ident(&arg.pat)?;
        (ident == name).then(|| (*arg.ty).clone())
    })
}

/// Function-scope `let mut` greens the portal loop does not rebind.
///
/// `warmspot.py ll_portal_runner` threads every green into the next
/// portal call. A loop-head `let` is recomputed on each iteration and
/// stays a body local; a mutable local that lives across the loop is an
/// input of the dispatch JitCode, after the portal-parameter greens, so
/// an inlined arm can move a new value into that register.
fn loop_carried_greens(
    config: &JitInterpConfig,
    func: &ItemFn,
) -> Vec<(Ident, jitcode_lower::ValueKind)> {
    let Some(portal) = codegen_trace::find_portal_loop(&func.block) else {
        return Vec::new();
    };
    let mut prefix_shadows = Vec::new();
    for stmt in &portal.body.stmts {
        if jitcode_lower::is_jit_merge_point_macro(stmt) {
            break;
        }
        let syn::Stmt::Local(local) = stmt else {
            continue;
        };
        if let Some((ident, _, _)) = local_pat_ident(&local.pat) {
            prefix_shadows.push(ident.to_string());
        }
    }
    let portal_names: Vec<String> = portal_green_params(config, func)
        .into_iter()
        .map(|(name, _, _)| name.to_string())
        .collect();
    let mut carried = Vec::new();
    for (index, stmt) in func.block.stmts.iter().enumerate() {
        if index == portal.index {
            break;
        }
        let syn::Stmt::Local(local) = stmt else {
            continue;
        };
        let Some((ident, is_mut, _)) = local_pat_ident(&local.pat) else {
            continue;
        };
        if !is_mut {
            continue;
        }
        let name = ident.to_string();
        if is_dispatch_entry_green(&name) {
            continue;
        }
        if prefix_shadows.iter().any(|shadow| shadow == &name) {
            continue;
        }
        if portal_names.iter().any(|param| param == &name) {
            continue;
        }
        let Some((green_index, expr)) = config
            .greens
            .iter()
            .enumerate()
            .find(|(_, green)| expr_single_ident(green).as_deref() == Some(name.as_str()))
        else {
            continue;
        };
        let tag = config.green_type_tags.get(green_index).copied().flatten();
        let kind = classify_green_kind(expr, tag, func);
        if let Some(slot) = carried.iter().position(|(id, _)| id == name.as_str()) {
            carried.remove(slot);
        }
        carried.push((ident.clone(), kind));
    }
    carried
}

fn param_scope_binds(sig: &syn::Signature) -> Vec<ScopeBind> {
    let mut binds = Vec::new();
    for arg in &sig.inputs {
        let syn::FnArg::Typed(arg) = arg else {
            continue;
        };
        let Some((ident, is_mut, _)) = local_pat_ident(&arg.pat) else {
            continue;
        };
        binds.push(ScopeBind {
            name: ident.to_string(),
            is_mut,
            ty: Some((*arg.ty).clone()),
            init: None,
        });
    }
    binds
}

fn scope_bind_from_local(local: &syn::Local) -> Option<ScopeBind> {
    let (ident, is_mut, ty) = local_pat_ident(&local.pat)?;
    Some(ScopeBind {
        name: ident.to_string(),
        is_mut,
        ty: ty.cloned(),
        init: local.init.as_ref().map(|init| (*init.expr).clone()),
    })
}

fn lookup_scope_bind<'a>(scopes: &'a [Vec<ScopeBind>], name: &str) -> Option<&'a ScopeBind> {
    scopes
        .iter()
        .rev()
        .find_map(|scope| scope.iter().rev().find(|bind| bind.name == name))
}

fn binding_value_kind(
    scopes: &[Vec<ScopeBind>],
    name: &str,
    tag: Option<green_type_tag::GreenTypeTag>,
) -> jitcode_lower::ValueKind {
    use jitcode_lower::ValueKind;
    if let Some(tag) = tag {
        return value_kind_from_tag(tag);
    }
    if let Some(bind) = lookup_scope_bind(scopes, name) {
        if let Some(ty) = &bind.ty {
            return value_kind_from_type(ty);
        }
        if let Some(init) = &bind.init
            && let Some(kind) = value_kind_from_expr(init)
        {
            return kind;
        }
    }
    ValueKind::Int
}

fn same_green_expr(left: &Expr, right: &Expr) -> bool {
    match (left, right) {
        (Expr::Path(left), Expr::Path(right)) => {
            left.path.get_ident().is_some() && left.path.get_ident() == right.path.get_ident()
        }
        _ => quote!(#left).to_string() == quote!(#right).to_string(),
    }
}

/// Assign loop-local greens from `ContinueRunningNormallyArgs` banks.
///
/// Declaration order, one index per bank. The marker's pc expression is
/// already written from `resume_pc` after the finish check, so a
/// negative green is not cast into that local first. Ref greens stay
/// out: a reference is not rebuilt from bits (`let mut` ref greens are
/// a `compile_error!` at expansion). An immutable `let` is skipped so a
/// loop-head binding is not assigned. A short bank is a decode fault
/// (`warmspot.py handle_jitexception` `getattr(e, attrname)[count]`).
fn green_local_assigns(
    scopes: &[Vec<ScopeBind>],
    greens: &[Expr],
    tags: &[Option<green_type_tag::GreenTypeTag>],
    pc_expr: &Expr,
) -> TokenStream {
    use jitcode_lower::ValueKind;
    let mut int_i = 0usize;
    let mut ref_i = 0usize;
    let mut float_i = 0usize;
    let mut assigns = Vec::new();
    for (index, green) in greens.iter().enumerate() {
        let tag = tags.get(index).copied().flatten();
        let kind = expr_single_ident(green)
            .map(|name| binding_value_kind(scopes, &name, tag))
            .unwrap_or_else(|| tag.map(value_kind_from_tag).unwrap_or(ValueKind::Int));
        let (field, bank_index) = match kind {
            ValueKind::Int => {
                let bank_index = int_i;
                int_i += 1;
                (quote!(green_int), bank_index)
            }
            ValueKind::Ref => {
                let bank_index = ref_i;
                ref_i += 1;
                (quote!(green_ref), bank_index)
            }
            ValueKind::Float => {
                let bank_index = float_i;
                float_i += 1;
                (quote!(green_float), bank_index)
            }
        };
        if same_green_expr(green, pc_expr) || matches!(kind, ValueKind::Ref) {
            continue;
        }
        let Expr::Path(path) = green else {
            continue;
        };
        let Some(ident) = path.path.get_ident() else {
            continue;
        };
        let Some(bind) = lookup_scope_bind(scopes, &ident.to_string()) else {
            continue;
        };
        if !bind.is_mut {
            continue;
        }
        assigns.push(quote! {
            let __bits = *__crn_args
                .#field
                .get(#bank_index)
                .expect("ContinueRunningNormally bank is short (warmspot.py handle_jitexception getattr(e, attrname)[count])");
            #ident = <_ as majit_metainterp::GreenFromI64>::from_green_i64(__bits);
        });
    }
    quote! { #(#assigns)* }
}

fn generate_merge_wrapper(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);
    let trace_fn_name = quote::format_ident!("__trace_{}", fn_name);
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let portal_greens = portal_green_params(config, func);
    let portal_green_names: Vec<_> = portal_greens.iter().map(|(name, _, _)| name).collect();
    let portal_green_param_decls: Vec<_> = portal_greens
        .iter()
        .map(|(name, ty, _)| quote! { #name: #ty, })
        .collect();
    let portal_green_call_args: Vec<_> = portal_green_names
        .iter()
        .map(|name| quote! { #name, })
        .collect();
    let carried_greens = loop_carried_greens(config, func);
    let carried_ref_mut_errors: Vec<_> = carried_greens
        .iter()
        .filter(|(_, kind)| matches!(kind, jitcode_lower::ValueKind::Ref))
        .map(|(name, _)| {
            let msg = format!(
                "let mut ref loop-carried green `{name}` cannot be written back \
                 from ContinueRunningNormally; use an immutable shared reference"
            );
            quote! { compile_error!(#msg); }
        })
        .collect();
    let carried_param_decls: Vec<_> = carried_greens
        .iter()
        .map(|(name, _)| {
            quote! { #name: impl ::core::marker::Copy + majit_ir::GreenAsI64, }
        })
        .collect();
    let carried_call_args: Vec<_> = carried_greens
        .iter()
        .map(|(name, _)| quote! { #name, })
        .collect();
    quote! {
        #[cold]
        #[inline(never)]
        #[allow(non_snake_case)]
        fn #merge_fn_name(
            __driver: &mut majit_metainterp::JitDriver<#state_type>,
            __env: &#env_type,
            __pc: usize,
            state: &#state_type,
            #(#portal_green_param_decls)*
            #(#carried_param_decls)*
        ) -> ::core::option::Option<(
            usize,
            ::std::vec::Vec<majit_metainterp::Value>,
            majit_metainterp::ContinueRunningNormallyArgs,
        )> {
            #(#carried_ref_mut_errors)*
            // Clone the dispatch JitCode Arc before the mutable
            // `merge_point` borrow so the closure can forward it to
            // `#trace_fn_name` without holding a `JitDriver` reference.
            let __dispatch_jitcode: Option<::std::sync::Arc<majit_metainterp::JitCode>> =
                __driver.dispatch_jitcode().cloned();
            __driver.merge_point(|__meta, __sym| {
                use majit_metainterp::JitCodeSym;
                // pyjitpl.py `self.pc = saved_pc`: a merge point whose
                // `reached_loop_header` returned without closing (the
                // `current_merge_points.append` path, :3058-3060) resumes the
                // walk at the merge point's own guest pc. This closure is
                // re-run for that, so the header-revisit fast path below —
                // which exists to close a trace that came back round to its
                // header — must not fire on the resumption: it would return
                // CloseLoop before the walk ran a single step, and the retrace
                // would compile an empty body.
                //
                // The flag is a PEEK. The dispatch merge-point arm is its
                // consumer (`take_merge_point_resumed`); taking it here would
                // let the resumed walk's own first merge-point visit close
                // instantly. The pc beside it is consumed here, because the pc
                // is what this closure hands to the walk.
                let (__resumed, __resume_pc) = match __meta.trace_ctx() {
                    ::core::option::Option::Some(__ctx) => {
                        (__ctx.merge_point_resumed, __ctx.walk_resume_pc.take())
                    }
                    ::core::option::Option::None => (false, ::core::option::Option::None),
                };
                // The position half of `self.pc = saved_pc`. `__pc` is the
                // native interpreter's `pc` captured by this closure, so it
                // names where the walk STARTED; the driver re-enters without
                // advancing it, and a walk that closed deeper in its own
                // segment would restart before the state it is carrying.
                // Re-seed at the pc the close published; `None` — every
                // re-entry that publishes none — keeps `__pc`.
                let __pc: usize = match __resume_pc {
                    ::core::option::Option::Some(__p) if __resumed => __p,
                    _ => __pc,
                };
                if !__resumed && __sym.trace_started && __pc == __sym.loop_header_pc() {
                    // `bhimpl_jit_merge_point` reads the merge-point
                    // registers just reached. Re-read those slots (live
                    // `MetaInterp.framestack`, or the Continue-walk
                    // snapshot taken before the last portal frame was
                    // popped) and build `live_arg_boxes` the way
                    // `reached_loop_header` does at `opimpl_jit_merge_point`.
                    __meta.close_header_revisit(__pc);
                    return majit_metainterp::TraceAction::CloseLoop;
                }
                // Slice X-D production wire-up: split-borrow the active
                // TraceCtx and a `jitcell_token_by_number` resolver so
                // the dispatcher's BC_CALL_ASSEMBLER_* path can route
                // through the production `Arc<JitCellToken>` rather
                // than the synth-Arc `_by_number_typed` fallback.  The
                // helper also hands over the recursive-call recursive-call seams
                // (green-key target resolver, inline decision, and the
                // `execute_token_raw` concrete executor) wired to the
                // production warmstate / backend.
                let __result = __meta
                    .with_trace_ctx_and_token_resolver(
                        |__ctx,
                         __resolve,
                         __rec_target,
                         __rec_decision,
                         __rec_exec,
                         __rec_exec_ref,
                         __rec_exec_float,
                         __rec_exec_void| {
                            let __runtime =
                                majit_metainterp::ClosureRuntimeWithResolver::new(
                                    |pc: usize| pc,
                                    __resolve,
                                    __rec_target,
                                    __rec_decision,
                                    __rec_exec,
                                    __rec_exec_ref,
                                    __rec_exec_float,
                                    __rec_exec_void,
                                );
                            #trace_fn_name(
                                __ctx,
                                __sym,
                                state,
                                __env,
                                __pc,
                                #(#portal_green_call_args)*
                                #(#carried_call_args)*
                                &__runtime,
                                __dispatch_jitcode.as_ref(),
                            )
                        },
                    )
                    .expect("merge_point invariant: tracing must be Some");
                __sym.trace_started = true;
                // pyjitpl.py blackhole_if_trace_too_long — check
                // AFTER executing the step (RPython _interpret loop order).
                let __too_long = __meta
                    .trace_ctx()
                    .map(|__ctx| __ctx.is_too_long())
                    .unwrap_or(false);
                if __too_long {
                    majit_metainterp::debug::log_one("jit-abort", "trace too long, aborting");
                    return majit_metainterp::TraceAction::Abort;
                }
                __result
            });
            // Single-pass tracing: surface the walk-final (pc, reds, green
            // banks) snapshot the CloseLoop arm stashed onto the MetaInterp
            // before draining the ctx. The banks are
            // `TraceCtx::portal_resume_args`, captured before
            // `compile_loop` clears it. `None` outside single-pass.
            __driver.take_single_pass_outcome()
        }
    }
}

/// Transform the original function: replace jit_merge_point!() and can_enter_jit!() markers.
/// Concrete (non-trace) path: rewrite `recursive_portal_call!` onto the
/// declared `recursive_entry`. `jit_merge_point!` / `can_enter_jit!` stay
/// as the exported no-ops.
fn rewrite_recursive_portal_calls(block: &mut syn::Block, recursive_entry: Option<&syn::Path>) {
    use syn::visit_mut::VisitMut;
    struct Visitor<'a> {
        recursive_entry: Option<&'a syn::Path>,
    }
    impl VisitMut for Visitor<'_> {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            if let syn::Stmt::Macro(sm) = stmt {
                let mut expr = Expr::Macro(syn::ExprMacro {
                    attrs: sm.attrs.clone(),
                    mac: sm.mac.clone(),
                });
                self.visit_expr_mut(&mut expr);
                if !matches!(expr, Expr::Macro(_)) {
                    *stmt = syn::Stmt::Expr(expr, sm.semi_token);
                    return;
                }
            }
            syn::visit_mut::visit_stmt_mut(self, stmt);
        }

        fn visit_expr_mut(&mut self, expr: &mut Expr) {
            syn::visit_mut::visit_expr_mut(self, expr);
            let Expr::Macro(em) = expr else {
                return;
            };
            let path_str = em
                .mac
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if path_str != "recursive_portal_call" && !path_str.ends_with("::recursive_portal_call")
            {
                return;
            }
            let args = em
                .mac
                .parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
                .expect("recursive_portal_call! takes `driver, green0, green1, ...`");
            let mut iter = args.into_iter();
            let _driver = iter
                .next()
                .expect("recursive_portal_call! requires a driver argument");
            let greens: Vec<Expr> = iter.collect();
            let entry = self.recursive_entry.unwrap_or_else(|| {
                panic!(
                    "recursive_portal_call! used but `#[jit_interp(..)]` declares no \
                     `recursive_entry = <fn path>` for the concrete fallback"
                )
            });
            let new_tokens = quote! { #entry(#(#greens),*) };
            *expr = syn::parse2(new_tokens)
                .expect("failed to parse recursive_portal_call concrete fallback");
        }
    }
    Visitor { recursive_entry }.visit_block_mut(block);
}

fn transform_function(config: &JitInterpConfig, func: &ItemFn, trace: bool) -> TokenStream {
    use syn::visit_mut::VisitMut;

    let vis = &func.vis;
    let sig = &func.sig;
    let attrs = &func.attrs;
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);

    // A field access through a `ref(T)` state scalar — `state.<ref>.<member>` —
    // lowers to a `getfield_gc_*` / `setfield_gc_*` on the JIT side, but the
    // concrete (non-JIT) function carries the ref as raw pointer bits (`usize`),
    // so the same source has no `.<member>` to resolve.  Rewrite each such
    // access in the concrete body to an unsafe deref of the raw pointer through
    // the declared `ref(T)` struct type, matching the heap object the JIT
    // getfield/setfield touches.
    struct RefFieldRewriter {
        // ref-scalar field name -> the `ref(T)` struct path `T`.
        ref_fields: std::collections::HashMap<String, syn::Path>,
        // `ref_fields` config entries: "StructLast::field" → pointee Path.
        // Used to determine which fields are ref-kind and propagate struct
        // type to local bindings.
        field_pointees: std::collections::HashMap<String, syn::Path>,
        // Local variable name -> struct type it points to (for ref bindings
        // returned from getfield_gc_r on state ref scalars or other locals).
        local_ref_types: std::collections::HashMap<String, syn::Path>,
        // `call_returns` entries: canonical func path segments → return struct type.
        // When `let x = func(...)` and func is in this map, `x` is recorded
        // as a local ref binding pointing to the declared struct type.
        call_returns: std::collections::HashMap<Vec<String>, syn::Path>,
        // `struct_allocs` entries: struct type segments → allocator function path.
        // When `let x = StructType { f0, f1 }` and StructType is in this map,
        // the concrete path rewrites the struct literal to `allocator(f0, f1)`.
        struct_allocs: std::collections::HashMap<Vec<String>, syn::Path>,
        // `inlined_prefix` entries: outer struct segments → (base field,
        // base path).  Empty unless a caller declares one.
        inlined_prefix: std::collections::HashMap<Vec<String>, (String, syn::Path)>,
        // Every `"StructLast::field"` the three field vocabularies declare.
        // The JIT lowerer consults the maps themselves; this walk only needs
        // to know whether a struct declares a field, not what it declared.
        declared_field_keys: std::collections::HashSet<String>,
        // `array_fields` entries: `"StructLast::field"` -> element type. The
        // field holds the buffer BASE POINTER, so an indexed access derefs
        // through it instead of reading the field itself.
        array_field_elems: std::collections::HashMap<String, syn::Path>,
        // `Element in Header` entries, same key. Element 0 is `Header.items`.
        array_headers: std::collections::HashMap<String, syn::Path>,
    }
    impl RefFieldRewriter {
        // For `e == state.<ref_scalar>`, return the `ref(T)` struct path.
        fn ref_struct_of_base(&self, e: &Expr) -> Option<syn::Path> {
            let Expr::Field(f) = e else { return None };
            if !matches!(&*f.base, Expr::Path(p) if p.path.is_ident("state")) {
                return None;
            }
            let syn::Member::Named(ref_name) = &f.member else {
                return None;
            };
            self.ref_fields.get(&ref_name.to_string()).cloned()
        }

        // For `e == <local_ident>`, return the struct type if it's a known
        // local ref binding.
        fn local_ref_struct_of_base(&self, e: &Expr) -> Option<syn::Path> {
            let Expr::Path(p) = e else { return None };
            let ident = p.path.get_ident()?;
            self.local_ref_types.get(&ident.to_string()).cloned()
        }

        // Given a struct path and a field name, check if the field is
        // declared as ref-kind in `field_pointees`.  Returns the pointee
        // type if so.
        fn field_pointee(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let struct_last = struct_path.segments.last()?.ident.to_string();
            let key = format!("{}::{}", struct_last, field_name);
            self.field_pointees.get(&key).cloned()
        }

        // The struct that declares `field_name`, so the emitted cast names
        // the type the field really lives on.  The JIT lowerer redirects the
        // same way; the two walks must agree, or this one dereferences a
        // struct that has no such field.
        fn declaring_struct(&self, struct_path: &syn::Path, field_name: &str) -> syn::Path {
            declaring_struct_of(
                &self.inlined_prefix,
                |path, field| {
                    path.segments.last().is_some_and(|last| {
                        self.declared_field_keys
                            .contains(&format!("{}::{}", last.ident, field))
                    })
                },
                struct_path,
                field_name,
            )
        }

        // Whether `struct_path::field_name` is an array base, and if so the
        // element type.
        fn array_field_key(struct_path: &syn::Path, field_name: &str) -> Option<String> {
            let struct_last = struct_path.segments.last()?.ident.to_string();
            Some(format!("{}::{}", struct_last, field_name))
        }

        fn array_field_elem(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let key = Self::array_field_key(struct_path, field_name)?;
            self.array_field_elems.get(&key).cloned()
        }

        fn array_header(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let key = Self::array_field_key(struct_path, field_name)?;
            self.array_headers.get(&key).cloned()
        }

        // Record a local binding as pointing to a struct type.
        fn record_local_ref(&mut self, name: &str, struct_type: syn::Path) {
            self.local_ref_types.insert(name.to_string(), struct_type);
        }

        // Check if a call expression's function is in `call_returns`.
        fn call_return_type(&self, func: &Expr) -> Option<syn::Path> {
            let Expr::Path(p) = func else { return None };
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            self.call_returns.get(&segs).cloned()
        }
    }
    impl VisitMut for RefFieldRewriter {
        fn visit_expr_mut(&mut self, expr: &mut Expr) {
            // Array element WRITE: `<base>.<array_field>[<idx>] = <rhs>`.
            //
            // Must precede the plain-field arms below: the field holds the
            // buffer BASE POINTER, so letting them rewrite `<base>.<field>` on
            // its own leaves a raw pointer being indexed with `[]`, which does
            // not compile. `jit_inline` intercepts the same shape in the same
            // order.
            if let Expr::Assign(assign) = expr
                && let Expr::Index(index_expr) = &*assign.left
                && let Expr::Field(field) = &*index_expr.expr
                && let Some(base_path) = self
                    .ref_struct_of_base(&field.base)
                    .or_else(|| self.local_ref_struct_of_base(&field.base))
                && let syn::Member::Named(member_id) = &field.member
                // Resolve against the struct that DECLARES the field, the way
                // `match_array_field_base` does on the JIT side. Looking the
                // element type up under the base's own spelling misses a field
                // an embedded base declares, and the plain-field arm below then
                // rewrites `<base>.<field>` to the buffer pointer and leaves the
                // `[]` on it, which does not compile.
                && let struct_path = self.declaring_struct(&base_path, &member_id.to_string())
                && self
                    .array_field_elem(&struct_path, &member_id.to_string())
                    .is_some()
            {
                let base = (*field.base).clone();
                let member = field.member.clone();
                let member_name = member_id.to_string();
                let header = self.array_header(&struct_path, &member_name);
                let element = self
                    .array_field_elem(&struct_path, &member_name)
                    .expect("array field element");
                let mut idx = (*index_expr.index).clone();
                let mut rhs = (*assign.right).clone();
                self.visit_expr_mut(&mut idx);
                self.visit_expr_mut(&mut rhs);
                // The assigned value is evaluated before the assignee place, so
                // an index or RHS with side effects must not be reordered:
                // `base.data[next_index()] = compute_value()` runs
                // `compute_value()` first. The JIT side lowers the right-hand
                // side first for the same reason, and the two paths have to
                // agree or a warm run observes a different order from a cold one.
                *expr = if let Some(header) = header {
                    syn::parse_quote! {
                        {
                            let __majit_arr_val = #rhs;
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                let __majit_arr_block = core::mem::transmute::<
                                    _,
                                    *mut #header,
                                >(
                                    (*(__majit_arr_obj as *const #struct_path)).#member,
                                );
                                let __majit_arr_items = (__majit_arr_block as *mut u8).add(
                                    core::mem::offset_of!(#header, items),
                                ) as *mut #element;
                                *__majit_arr_items.add(__majit_arr_idx as usize) = __majit_arr_val;
                            }
                        }
                    }
                } else {
                    syn::parse_quote! {
                        {
                            let __majit_arr_val = #rhs;
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                *((*(__majit_arr_obj as *mut #struct_path))
                                    .#member
                                    .add(__majit_arr_idx as usize)) = __majit_arr_val;
                            }
                        }
                    }
                };
                return;
            }

            // Array element READ: `<base>.<array_field>[<idx>]`.
            if let Expr::Index(index_expr) = expr
                && let Expr::Field(field) = &*index_expr.expr
                && let Some(base_path) = self
                    .ref_struct_of_base(&field.base)
                    .or_else(|| self.local_ref_struct_of_base(&field.base))
                && let syn::Member::Named(member_id) = &field.member
                && let struct_path = self.declaring_struct(&base_path, &member_id.to_string())
                && self
                    .array_field_elem(&struct_path, &member_id.to_string())
                    .is_some()
            {
                let base = (*field.base).clone();
                let member = field.member.clone();
                let member_name = member_id.to_string();
                let header = self.array_header(&struct_path, &member_name);
                let element = self
                    .array_field_elem(&struct_path, &member_name)
                    .expect("array field element");
                let mut idx = (*index_expr.index).clone();
                self.visit_expr_mut(&mut idx);
                *expr = if let Some(header) = header {
                    syn::parse_quote! {
                        {
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                let __majit_arr_block = core::mem::transmute::<
                                    _,
                                    *mut #header,
                                >(
                                    (*(__majit_arr_obj as *const #struct_path)).#member,
                                );
                                let __majit_arr_items = (__majit_arr_block as *mut u8).add(
                                    core::mem::offset_of!(#header, items),
                                ) as *mut #element;
                                *__majit_arr_items.add(__majit_arr_idx as usize)
                            }
                        }
                    }
                } else {
                    syn::parse_quote! {
                        {
                            let __majit_arr_obj = #base;
                            let __majit_arr_idx = #idx;
                            unsafe {
                                *((*(__majit_arr_obj as *const #struct_path))
                                    .#member
                                    .add(__majit_arr_idx as usize))
                            }
                        }
                    }
                };
                return;
            }

            // Write-through: `state.<ref>.<member> = <rhs>` ->
            // `unsafe { (*(state.<ref> as *mut T)).<member> = <rhs> }`.
            if let Expr::Assign(assign) = expr
                && let Expr::Field(lhs) = &*assign.left
            {
                // Unify state-ref and local-ref write-through.
                let base_struct = self
                    .ref_struct_of_base(&lhs.base)
                    .or_else(|| self.local_ref_struct_of_base(&lhs.base));
                if let Some(struct_path) = base_struct {
                    let base = (*lhs.base).clone();
                    let member = lhs.member.clone();
                    let member_name = match &lhs.member {
                        syn::Member::Named(id) => id.to_string(),
                        _ => String::new(),
                    };
                    // Resolve against the struct that declares the field, so
                    // the cast below names the type it really lives on.
                    let struct_path = self.declaring_struct(&struct_path, &member_name);
                    let mut rhs = (*assign.right).clone();
                    self.visit_expr_mut(&mut rhs);
                    if let Some(pointee) = self.field_pointee(&struct_path, &member_name) {
                        // Ref-kind field → cast RHS from usize to *mut Pointee.
                        *expr = syn::parse_quote! {
                            unsafe { (*(#base as *mut #struct_path)).#member = #rhs as *mut #pointee }
                        };
                    } else {
                        *expr = syn::parse_quote! {
                            unsafe { (*(#base as *mut #struct_path)).#member = #rhs }
                        };
                    }
                    return;
                }
            }
            // Read: `state.<ref>.<member>` reads a mutable heap field live from
            // the object pointer. The walk is the sole executor, so the field
            // holds the current value; the JIT side records BC_GETFIELD_GC_I.
            if let Expr::Field(field) = expr {
                if let Some(struct_path) = self.ref_struct_of_base(&field.base) {
                    let base = (*field.base).clone();
                    let member = field.member.clone();
                    let member_name = match &field.member {
                        syn::Member::Named(id) => id.to_string(),
                        _ => String::new(),
                    };
                    // Resolve against the struct that declares the field, so
                    // the cast below names the type it really lives on.
                    let struct_path = self.declaring_struct(&struct_path, &member_name);
                    let is_ref_field = self.field_pointee(&struct_path, &member_name).is_some();
                    if is_ref_field {
                        // Ref-kind field → cast to usize (JIT ref bank).
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member as usize
                                }
                            }
                        };
                    } else {
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member
                                }
                            }
                        };
                    }
                    return;
                }
                // Read on local ref binding: `<local>.<member>`.
                if let Some(struct_path) = self.local_ref_struct_of_base(&field.base) {
                    let base = (*field.base).clone();
                    let member = field.member.clone();
                    let member_name = match &field.member {
                        syn::Member::Named(id) => id.to_string(),
                        _ => String::new(),
                    };
                    // Resolve against the struct that declares the field, so
                    // the cast below names the type it really lives on.
                    let struct_path = self.declaring_struct(&struct_path, &member_name);
                    let is_ref_field = self.field_pointee(&struct_path, &member_name).is_some();
                    if is_ref_field {
                        // Ref-kind field → cast result to usize (JIT ref bank).
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member as usize
                                }
                            }
                        };
                    } else {
                        // Int-kind field → value used as-is.
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member
                                }
                            }
                        };
                    }
                    return;
                }
            }
            syn::visit_mut::visit_expr_mut(self, expr);
        }

        // Track local `let` bindings that produce ref-typed values from
        // ref field reads, so subsequent field access rewrites work.
        fn visit_local_mut(&mut self, local: &mut syn::Local) {
            // First, recurse into the init expression to rewrite any ref
            // field reads contained within.
            syn::visit_mut::visit_local_mut(self, local);

            // After rewriting, check if the init expression was a
            // state.<ref>.<member> or local.<member> read whose field is
            // ref-kind, and if so, record the local as a ref binding.
            //
            // The rewriter has already transformed the init expr, but we
            // can recover the original shape from the pattern: if the init
            // was `state.selected_ref.head` and `Stack::head => Node` is
            // in field_pointees, the local should be typed as `*mut Node`.
            //
            // Since VisitMut has already rewritten the init expr, we check
            // the LOCAL PATTERN instead and rely on a pre-pass that recorded
            // the binding name before rewriting.  We handle this via
            // visit_stmt_mut below.
        }

        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // Before the default visit rewrites the init expression, check
            // if this is a `let <ident> = <base>.<field>` where <base> is
            // a state ref scalar or a local ref binding, and the field is
            // declared ref-kind.
            if let syn::Stmt::Local(local) = stmt
                && let Some(init) = &local.init
            {
                if let Expr::Field(field) = &*init.expr {
                    let member_name = match &field.member {
                        syn::Member::Named(id) => Some(id.to_string()),
                        _ => None,
                    };
                    if let Some(member_name) = member_name {
                        // Check if base is state.<ref_scalar>
                        let struct_path = self
                            .ref_struct_of_base(&field.base)
                            .or_else(|| self.local_ref_struct_of_base(&field.base));
                        if let Some(struct_path) = struct_path
                            && let Some(pointee) = self.field_pointee(
                                &self.declaring_struct(&struct_path, &member_name),
                                &member_name,
                            )
                        {
                            // This field is ref-kind → the local is a
                            // ref binding pointing to pointee type.
                            if let syn::Pat::Ident(pat_ident) = &local.pat {
                                self.record_local_ref(&pat_ident.ident.to_string(), pointee);
                            }
                        }
                    }
                }
                // `let x = state.<ref_scalar>` → the local carries the
                // `ref(T)` declaration's own pointee. The arm above answers
                // the field-of-a-ref shape; this one is the ref itself, and
                // without it `state.sel.value` compiles while the same read
                // split across two statements does not.
                if let Some(struct_path) = self.ref_struct_of_base(&init.expr)
                    && let syn::Pat::Ident(pat_ident) = &local.pat
                {
                    self.record_local_ref(&pat_ident.ident.to_string(), struct_path);
                }
                // `let x = func(...)` where func is in `call_returns` →
                // record x as a local ref binding with the declared type.
                if let Expr::Call(call) = &*init.expr
                    && let Some(struct_type) = self.call_return_type(&call.func)
                    && let syn::Pat::Ident(pat_ident) = &local.pat
                {
                    self.record_local_ref(&pat_ident.ident.to_string(), struct_type);
                }
            }
            // Rewrite struct literal inits on the concrete path:
            // `let x = StructType { f0: v0, f1: v1 }` where StructType is
            // in `struct_allocs` → `let x = allocator_func(v0, v1)`.
            // The JIT path handles struct literals via `lower_struct_value`
            // (New + SetfieldGc); this rewrite makes the concrete path
            // produce the same allocation through the runtime allocator.
            if let syn::Stmt::Local(local) = stmt
                && let Some(init) = &mut local.init
                && let Expr::Struct(s) = &*init.expr
            {
                let segs: Vec<String> = s
                    .path
                    .segments
                    .iter()
                    .map(|seg| seg.ident.to_string())
                    .collect();
                if let Some(alloc_func) = self.struct_allocs.get(&segs).cloned() {
                    let field_args: Vec<Expr> = s.fields.iter().map(|f| f.expr.clone()).collect();
                    *init.expr = syn::parse_quote! {
                        #alloc_func(#(#field_args),*)
                    };
                }
            }
            // Now run the default visitor which will rewrite the init expr.
            syn::visit_mut::visit_stmt_mut(self, stmt);
        }
    }

    let mut block = func.block.clone();
    if let Some(sf) = &config.state_fields {
        let ref_fields: std::collections::HashMap<String, syn::Path> = sf
            .fields
            .iter()
            .filter_map(|f| match &f.kind {
                StateFieldKind::Ref(p) => Some((f.name.to_string(), p.clone())),
                _ => None,
            })
            .collect();
        if !ref_fields.is_empty() {
            // Build field_pointees map from the macro's `ref_fields` config.
            let field_pointees: std::collections::HashMap<String, syn::Path> = config
                .ref_fields
                .iter()
                .map(|entry| {
                    let struct_last = entry
                        .struct_type
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default();
                    let key = format!("{}::{}", struct_last, entry.field);
                    (key, entry.pointee_type.clone())
                })
                .collect();
            let call_returns_map: std::collections::HashMap<Vec<String>, syn::Path> = config
                .call_returns
                .iter()
                .map(|(func, ret_type)| {
                    let segs: Vec<String> =
                        func.segments.iter().map(|s| s.ident.to_string()).collect();
                    (segs, ret_type.clone())
                })
                .chain(config.pool_arrays.iter().filter_map(|entry| {
                    let ret_type = entry.element_type.clone()?;
                    let segs: Vec<String> = entry
                        .getter
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect();
                    Some((segs, ret_type))
                }))
                .collect();
            let struct_allocs_map: std::collections::HashMap<Vec<String>, syn::Path> = config
                .struct_allocs
                .iter()
                .map(|(struct_path, alloc_func)| {
                    let segs: Vec<String> = struct_path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect();
                    (segs, alloc_func.clone())
                })
                .collect();
            RefFieldRewriter {
                ref_fields,
                declared_field_keys: declared_field_keys(
                    &config.ref_fields,
                    &config.int_fields,
                    &config.array_fields,
                ),
                array_field_elems: config
                    .array_fields
                    .iter()
                    .filter_map(|e| {
                        let last = e.struct_type.segments.last()?.ident.to_string();
                        Some((format!("{}::{}", last, e.field), e.element_type.clone()))
                    })
                    .collect(),
                array_headers: config
                    .array_fields
                    .iter()
                    .filter_map(|e| {
                        let header = e.header.clone()?;
                        let last = e.struct_type.segments.last()?.ident.to_string();
                        Some((format!("{}::{}", last, e.field), header))
                    })
                    .collect(),
                field_pointees,
                local_ref_types: std::collections::HashMap::new(),
                call_returns: call_returns_map,
                struct_allocs: struct_allocs_map,
                inlined_prefix: inlined_prefix_index(&config.inlined_prefix),
            }
            .visit_block_mut(&mut block);
        }
    }

    // Rewrite the function body, replacing marker macros
    let finish_return = finish_return_for(&sig.output);
    let portal_green_args: Vec<Expr> = portal_green_params(config, func)
        .into_iter()
        .map(|(name, _, _)| syn::parse_quote!(#name))
        .collect();
    let carried_green_args: Vec<Expr> = loop_carried_greens(config, func)
        .into_iter()
        .map(|(name, _)| syn::parse_quote!(#name))
        .collect();
    // `warmspot.py ll_portal_runner` is a function in front of the portal
    // body. The split is only sound when the door can run from the parameters
    // alone and the traced loop is the body's first statement: any local the
    // door reads is created inside the body, and the inline door stays there.
    let runner_shape =
        trace && finish_return.is_some() && portal_runner_shape(&func.sig, &block, &config.greens);
    let (body_stmts, runner_door) = if trace {
        rewrite_body(
            &block,
            &merge_fn_name,
            &portal_green_args,
            &carried_green_args,
            &config.greens,
            config.greens_declared,
            &config.green_type_tags,
            config.recursive_entry.as_ref(),
            finish_return.as_ref(),
            !runner_shape,
            sig,
        )
    } else {
        let mut block = block;
        rewrite_recursive_portal_calls(&mut block, config.recursive_entry.as_ref());
        (block.stmts, None)
    };

    if runner_shape {
        let door = runner_door.expect("portal runner shape has a function-entry door");
        let finish = finish_return
            .as_ref()
            .expect("portal runner shape has a finish projection");
        emit_portal_runner(vis, sig, attrs, &body_stmts, door, finish)
    } else {
        quote! {
            #(#attrs)*
            #vis #sig {
                #(#body_stmts)*
            }
        }
    }
}

/// `ll_portal_runner` in front of `__<name>_portal`. The original name,
/// attributes and visibility stay on the runner. Registrations
/// (`__dispatch_jitcode_*`, the jitcode name, `__merge_*` / `__trace_*`)
/// keep that same identifier; the body is only the call target.
fn emit_portal_runner(
    vis: &syn::Visibility,
    sig: &syn::Signature,
    attrs: &[syn::Attribute],
    body_stmts: &[syn::Stmt],
    door: RunnerDoor,
    finish: &FinishReturn,
) -> TokenStream {
    let portal_ident = quote::format_ident!("__{}_portal", sig.ident);
    let mut portal_sig = sig.clone();
    portal_sig.ident = portal_ident.clone();
    let params = fn_param_idents(sig).expect("portal runner parameters are plain idents");
    let pc_ident = bare_ident(&door.pc).expect("portal runner pc is a parameter");
    let run_args: Vec<TokenStream> = params.iter().map(|id| quote! { #id }).collect();
    let resume_args: Vec<TokenStream> = params
        .iter()
        .map(|id| {
            if id == &pc_ident {
                quote! { __resume_pc }
            } else {
                quote! { #id }
            }
        })
        .collect();
    let direct = finish.return_finished_word();
    let drain = finish.drain(&door.driver);
    let single_pass_drain = finish.drain_single_pass(&door.driver);
    let epilogue = &body_stmts[1..];
    let key = &door.key;
    let driver = &door.driver;
    let pc = &door.pc;
    let state = &door.state;
    let env = &door.env;
    quote! {
        #(#attrs)*
        #vis #sig {
            // `warmspot.py ll_portal_runner`: maybe_compile_and_run, and on
            // `execute_assembler`'s DoneWithThisFrame fast path return without
            // entering the portal body.
            {
                let (__green_hash, __make_key) = #key;
                match #driver.function_entry_runner(
                    __green_hash,
                    __make_key,
                    #pc,
                    &mut #state,
                    #env,
                ) {
                    majit_metainterp::FunctionEntryRunner::Finished(__finish_word) => {
                        #direct
                    }
                    majit_metainterp::FunctionEntryRunner::Resume(__resume_pc) => {
                        #drain
                        // A guard-resume bridge that reaches the portal return
                        // publishes on the single-pass latch and reports
                        // `usize::MAX`. `warmspot.py ll_portal_runner` returns
                        // `DoneWithThisFrameRef.result` whether
                        // `execute_assembler`, `blackhole.py
                        // _done_with_this_frame`, or `pyjitpl.py
                        // MetaInterp.compile_done_with_this_frame` raised it.
                        if #driver.take_single_pass_finish() {
                            #single_pass_drain
                        }
                        if __resume_pc != usize::MAX {
                            return #portal_ident(#(#resume_args),*);
                        }
                    }
                    majit_metainterp::FunctionEntryRunner::Run => {
                        #drain
                        // Same single-pass latch as `Resume`: a bridge that
                        // reaches the portal return publishes there and
                        // reports no resume pc, so this arm is `Run`.
                        if #driver.take_single_pass_finish() {
                            #single_pass_drain
                        }
                        return #portal_ident(#(#run_args),*);
                    }
                }
            }
            #(#epilogue)*
        }

        #[inline(never)]
        #[allow(non_snake_case)]
        #portal_sig {
            #(#body_stmts)*
        }
    }
}

struct RunnerDoor {
    key: TokenStream,
    driver: Expr,
    pc: Expr,
    state: Expr,
    env: Expr,
}

fn fn_param_idents(sig: &syn::Signature) -> Option<Vec<Ident>> {
    let mut out = Vec::new();
    for arg in &sig.inputs {
        let syn::FnArg::Typed(pat) = arg else {
            return None;
        };
        let syn::Pat::Ident(id) = &*pat.pat else {
            return None;
        };
        out.push(id.ident.clone());
    }
    Some(out)
}

fn bare_ident(expr: &Expr) -> Option<Ident> {
    match expr {
        Expr::Path(path)
            if path.qself.is_none()
                && path.attrs.is_empty()
                && path.path.segments.len() == 1
                && path.path.segments[0].arguments.is_none() =>
        {
            Some(path.path.segments[0].ident.clone())
        }
        _ => None,
    }
}

/// Root ident of a door operand: the parameter, or the parameter under `*` / `&`.
fn expr_root_ident(expr: &Expr) -> Option<Ident> {
    match expr {
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            expr_root_ident(&unary.expr)
        }
        Expr::Reference(reference) => expr_root_ident(&reference.expr),
        _ => bare_ident(expr),
    }
}

/// The traced loop is the body's first statement, and driver/pc/state/env are
/// parameters, so `ll_portal_runner` can run before the portal body.
fn portal_runner_shape(sig: &syn::Signature, block: &syn::Block, greens: &[Expr]) -> bool {
    if greens.is_empty() {
        return false;
    }
    if traced_loop_stmt_index(&block.stmts) != Some(0) {
        return false;
    }
    let Some(params) = fn_param_idents(sig) else {
        return false;
    };
    let scan = scan_single_pass_close(block);
    let Some(args) = scan.first_state else {
        return false;
    };
    let driver = args
        .driver
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(driver));
    let env = args
        .env
        .clone()
        .unwrap_or_else(|| syn::parse_quote!(program));
    let pc = args.pc.clone().unwrap_or_else(|| syn::parse_quote!(pc));
    let Some(state) = args.state.clone() else {
        return false;
    };
    let is_param = |expr: &Expr| {
        expr_root_ident(expr).is_some_and(|id| params.iter().any(|param| param == &id))
    };
    is_param(&driver)
        && is_param(&env)
        && is_param(&state)
        && bare_ident(&pc).is_some_and(|id| params.iter().any(|param| param == &id))
}

/// How a compiled run that ended in FINISH is returned from the portal.
///
/// `compile.py` marks a FINISH descr `final_descr = True`: the traced
/// function has RETURNED, and upstream unwinds the portal by raising
/// `jitexc.DoneWithThisFrame*`. `back_edge*` reports `Option<resume_pc>`, which
/// has no variant for that, so the result travels out of band in the driver
/// (`take_back_edge_finish_*`) and the expansion turns it back into the portal's
/// own `return`. Which projection applies is decided by the portal's declared
/// return type.
#[derive(Clone, Copy)]
enum FinishReturnKind {
    Int,
    Float,
    /// `*mut T` / `*const T`. `warmstate.py execute_assembler` returns
    /// `DoneWithThisFrameDescrRef.get_result` for this portal.
    Ref,
}

#[derive(Clone)]
struct FinishReturn {
    kind: FinishReturnKind,
    /// The portal's return type, when it is not the projection's own word type
    /// (`i64` / `f64`) and therefore needs a cast. `None` means no cast.
    cast_to: Option<syn::Type>,
}

/// Classify a portal's return type into a FINISH projection, or `None` when the
/// expansion cannot build that type out of a `majit_ir::Value` (for example a
/// portal returning `String`). `None` leaves the back edge exactly as it was:
/// the driver still returns `Some(target_pc)` on FINISH for those portals.
fn finish_return_for(output: &syn::ReturnType) -> Option<FinishReturn> {
    let syn::ReturnType::Type(_, ty) = output else {
        return None;
    };
    // A raw pointer is a ref finish. The portal must spell `*mut T` or
    // `*const T`: a type alias is a path, and this macro cannot see through it.
    if let syn::Type::Ptr(_) = ty.as_ref() {
        return Some(FinishReturn {
            kind: FinishReturnKind::Ref,
            cast_to: Some(ty.as_ref().clone()),
        });
    }
    let syn::Type::Path(type_path) = ty.as_ref() else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    let segment = type_path.path.segments.last()?;
    if !segment.arguments.is_none() {
        return None;
    }
    let (kind, is_word_type) = match segment.ident.to_string().as_str() {
        "i64" => (FinishReturnKind::Int, true),
        "i8" | "i16" | "i32" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
            (FinishReturnKind::Int, false)
        }
        "f64" => (FinishReturnKind::Float, true),
        "f32" => (FinishReturnKind::Float, false),
        _ => return None,
    };
    Some(FinishReturn {
        kind,
        cast_to: (!is_word_type).then(|| ty.as_ref().clone()),
    })
}

impl FinishReturn {
    /// The statement that drains the driver's FINISH latch and returns it as the
    /// portal's result. Emitted immediately after the `back_edge*` call whose
    /// compiled run may have set it, and BEFORE that call's `Some(resume_pc)` is
    /// used: on FINISH that pc is the back edge, so resuming there would re-run
    /// the loop the compiled run already completed.
    fn drain(&self, driver_expr: &Expr) -> TokenStream {
        let take = match self.kind {
            FinishReturnKind::Int => quote! { take_back_edge_finish_int },
            FinishReturnKind::Float => quote! { take_back_edge_finish_float },
            FinishReturnKind::Ref => quote! { take_back_edge_finish_ref },
        };
        let returned = match &self.cast_to {
            Some(ty) => quote! { __finish_value as #ty },
            None => quote! { __finish_value },
        };
        quote! {
            if let Some(__finish_value) = #driver_expr.#take() {
                return #returned;
            }
        }
    }

    /// Drain the result of the tracing walk itself.  The dispatch JitCode has
    /// already executed the source post-loop epilogue; returning here avoids
    /// running that suffix again after the merge hook writes state back.
    fn drain_single_pass(&self, driver_expr: &Expr) -> TokenStream {
        let take = match self.kind {
            FinishReturnKind::Int => quote! { take_single_pass_finish_int },
            FinishReturnKind::Float => quote! { take_single_pass_finish_float },
            FinishReturnKind::Ref => quote! { take_single_pass_finish_ref },
        };
        let returned = match &self.cast_to {
            Some(ty) => quote! { __finish_value as #ty },
            None => quote! { __finish_value },
        };
        quote! {
            if let Some(__finish_value) = #driver_expr.#take() {
                return #returned;
            }
        }
    }

    /// `execute_assembler`'s fast path, returned from `ll_portal_runner`
    /// without the finish latch. `__finish_word` is the `i64` the raw poll
    /// produced (an int, or a ref address).
    fn return_finished_word(&self) -> TokenStream {
        match self.kind {
            FinishReturnKind::Int => match &self.cast_to {
                Some(ty) => quote! { return __finish_word as #ty; },
                None => quote! { return __finish_word; },
            },
            FinishReturnKind::Float => match &self.cast_to {
                Some(ty) => quote! { return (f64::from_bits(__finish_word as u64)) as #ty; },
                None => quote! { return f64::from_bits(__finish_word as u64); },
            },
            FinishReturnKind::Ref => {
                let ty = self
                    .cast_to
                    .clone()
                    .expect("a ref finish carries the portal's pointer type");
                quote! { return (__finish_word as usize) as #ty; }
            }
        }
    }
}

/// Parsed arguments of a `jit_merge_point!` invocation.
///
/// The bare form (`jit_merge_point!()`) is the observer/replay close.
/// `jit_merge_point!(driver, env, pc; state)` is the single-executor close:
/// the walk keeps executing, so the live struct and the trace shadow must
/// stay equal.
#[derive(Default, Clone)]
struct MergePointArgs {
    driver: Option<Expr>,
    env: Option<Expr>,
    pc: Option<Expr>,
    /// Single-pass tracing opt-in handle: the mutable native `JitState`
    /// binding. Supplied as the first expr after `;`
    /// (`jit_merge_point!(driver, env, pc; state)`). When present, the
    /// expansion emits the gated post-walk transfer hook; when absent
    /// (the default `jit_merge_point!()` form), the expansion is the
    /// byte-identical observer/replay statement.
    state: Option<Expr>,
}

impl Parse for MergePointArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self::default());
        }
        let driver: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let env: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let pc: Expr = input.parse()?;
        let mut state = None;
        if input.peek(Token![;]) {
            input.parse::<Token![;]>()?;
            let tail: Punctuated<Expr, Token![,]> =
                input.parse_terminated(Expr::parse, Token![,])?;
            // The first tail expr is the single-pass `state` handle; any
            // further exprs remain accepted-and-ignored (legacy form).
            state = tail.into_iter().next();
        }
        Ok(Self {
            driver: Some(driver),
            env: Some(env),
            pc: Some(pc),
            state,
        })
    }
}

/// Whether any `jit_merge_point!` in this body carries `; state`.
///
/// The two spans are what `validate_merge_point_forms` reports: a body is
/// allowed to spell the marker many times, but every one of them has to agree
/// on who owns the state between passes.
struct SinglePassCloseScan {
    found: bool,
    first_state: Option<MergePointArgs>,
    /// First marker carrying `; state` — the walk keeps executing the guest.
    single_pass_site: Option<proc_macro2::Span>,
    /// First marker without it — an outer executor keeps executing the guest.
    outer_owned_site: Option<proc_macro2::Span>,
}

impl<'ast> syn::visit::Visit<'ast> for SinglePassCloseScan {
    fn visit_macro(&mut self, mac: &'ast syn::Macro) {
        let path_str = mac
            .path
            .segments
            .iter()
            .map(|s| s.ident.to_string())
            .collect::<Vec<_>>()
            .join("::");
        if path_str == "jit_merge_point" || path_str.ends_with("::jit_merge_point") {
            let site = mac.path.segments[0].ident.span();
            let args = syn::parse2::<MergePointArgs>(mac.tokens.clone()).unwrap_or_default();
            if args.state.is_some() {
                self.found = true;
                if self.first_state.is_none() {
                    self.first_state = Some(args);
                }
                self.single_pass_site.get_or_insert(site);
            } else {
                self.outer_owned_site.get_or_insert(site);
            }
        }
        syn::visit::visit_macro(self, mac);
    }
}

fn scan_single_pass_close(block: &syn::Block) -> SinglePassCloseScan {
    let mut scan = SinglePassCloseScan {
        found: false,
        first_state: None,
        single_pass_site: None,
        outer_owned_site: None,
    };
    syn::visit::Visit::visit_block(&mut scan, block);
    scan
}

/// Whether any `jit_merge_point!` in this function carries `; state`.
///
/// `rewrite_body` uses `scan_single_pass_close` for the entry door;
/// `codegen_state::generate_jit_state` publishes the inverted answer on
/// `VirtualizableInfo::outer_executor_owns_state`.
fn has_single_pass_close(func: &ItemFn) -> bool {
    scan_single_pass_close(&func.block).found
}

/// Every `jit_merge_point!` in one body must agree on the merge-point form.
///
/// The form answers who keeps executing the guest across a pass: a marker
/// carrying `; state` closes the loop on the walk itself, while one without it
/// leaves an outer executor running the live struct. That answer is published
/// once per interpreter, on `VirtualizableInfo::outer_executor_owns_state`,
/// and `TraceCtx::synchronize_virtualizable` reads it to decide whether
/// flushing the trace shadow back into the native state is safe. One field
/// cannot carry two answers: a body mixing the forms would flush for whichever
/// form lost the vote, which for an outer-owned marker is exactly the clobber
/// the carve-out exists to prevent. Reject the mix here, where the author can
/// see it, rather than resolving it silently.
fn validate_merge_point_forms(func: &ItemFn) -> syn::Result<()> {
    let scan = scan_single_pass_close(&func.block);
    let (Some(outer_owned), Some(single_pass)) = (scan.outer_owned_site, scan.single_pass_site)
    else {
        return Ok(());
    };
    let mut err = syn::Error::new(
        single_pass,
        "this `jit_merge_point!` carries `; state`, so the walk closes the loop \
         and owns the state between passes",
    );
    err.combine(syn::Error::new(
        outer_owned,
        "this `jit_merge_point!` in the same function does not, so an outer \
         executor owns it -- one function declares one ownership mode, because \
         `VirtualizableInfo::outer_executor_owns_state` is published per \
         interpreter and decides whether the virtualizable write-back is a \
         clobber",
    ));
    Err(err)
}

/// Rewrite function body: replace jit_merge_point!() and can_enter_jit!() calls.
fn rewrite_body(
    block: &syn::Block,
    merge_fn_name: &Ident,
    portal_green_args: &[Expr],
    carried_green_args: &[Expr],
    default_greens: &[Expr],
    greens_declared: bool,
    default_green_type_tags: &[Option<green_type_tag::GreenTypeTag>],
    recursive_entry: Option<&Path>,
    finish_return: Option<&FinishReturn>,
    insert_entry_door: bool,
    func_sig: &syn::Signature,
) -> (Vec<syn::Stmt>, Option<RunnerDoor>) {
    use syn::visit_mut::VisitMut;

    struct CanEnterJitArgs {
        driver: Expr,
        target: Expr,
        state: Expr,
        env: Expr,
        pre_run: Expr,
        pc: Option<Expr>,
        stacksize: Option<Expr>,
        greens: Vec<Expr>,
    }

    impl Parse for CanEnterJitArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let target: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let state: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pre_run: Expr = input.parse()?;

            let mut pc = None;
            let mut stacksize = None;
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                pc = Some(input.parse::<Expr>()?);
                input.parse::<Token![,]>()?;
                stacksize = Some(input.parse::<Expr>()?);
            }

            let mut greens = Vec::new();
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let exprs: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
                greens = exprs.into_iter().collect();
            }

            Ok(Self {
                driver,
                target,
                state,
                env,
                pre_run,
                pc,
                stacksize,
                greens,
            })
        }
    }

    /// Emit a single `(i64, majit_ir::GreenType)` pair for one green
    /// expression, dispatching on the optional per-green type tag.
    /// Untagged (`None`) greens flow through the
    /// `<_ as GreenAsI64>::__green_repr` trait. Tagged greens emit
    /// explicit casts so
    /// `&str`-bearing greens carry `GreenType::Str` instead of being
    /// silently routed through the blanket `impl<T: ?Sized>` Ref impl.
    fn emit_green_repr(spec_expr: &Expr, tag: Option<green_type_tag::GreenTypeTag>) -> TokenStream {
        use green_type_tag::GreenTypeTag;
        match tag {
            // Untagged greens go through the `GreenAsI64` trait —
            // primitive types route to `GreenType::Int`, floats to
            // `GreenType::Float`, references to `GreenType::Ref`.  An
            // untagged `&str` lands on the blanket `impl<T: ?Sized>
            // GreenAsI64 for &T` and carries `GreenType::Ref` (raw
            // pointer identity).  Consumers wanting RPython STR /
            // UNICODE content semantics MUST tag explicitly via
            // `greens = [name: str]` / `greens = [name: unicode]`;
            // `&str` is ambiguous between `rstr.STR` (UTF-8 byte
            // string) and `rstr.UNICODE` (codepoint sequence) and the
            // macro cannot pick one without an explicit declaration.
            None => quote! {
                <_ as majit_ir::GreenAsI64>::__green_repr(#spec_expr)
            },
            Some(GreenTypeTag::Int) => quote! {
                ((#spec_expr) as i64, majit_ir::GreenType::Int)
            },
            Some(GreenTypeTag::Float) => quote! {
                {
                    let __green_f: f64 = (#spec_expr) as f64;
                    (__green_f.to_bits() as i64, majit_ir::GreenType::Float)
                }
            },
            Some(GreenTypeTag::Ref) => quote! {
                {
                    let (__green_bits, _) =
                        <_ as majit_ir::GreenAsI64>::__green_repr(#spec_expr);
                    (__green_bits, majit_ir::GreenType::Ref)
                }
            },
            // ABI: the i64 is the address of a `'static` slot holding
            // a `&'static str`.  `majit_ir::value::default_str_eq` /
            // `default_str_hash` / `default_unicode_hash` dereference
            // the i64 as `*const &'static str` and read the fat
            // pointer (data + len) — RPython's `rstr.STR*` /
            // `rstr.UNICODE*` carry their length internally; pyre
            // mirrors that contract by storing the fat pointer at a
            // stable address rather than the bare data pointer.
            //
            // TODO: allocation lifetime divergence
            // from RPython, intentionally deferred):
            //
            //   * RPython: `rstr.STR*` is GC-allocated *once per
            //     JitCell* and naturally stable for the JitCell's
            //     lifetime.  `JitCell.greenargs[i]` holds the rstr
            //     pointer; subsequent merge-point hits for the same
            //     `(jitdriver, greenkey)` pass the existing pointer
            //     into `equal_whatever` / `hash_whatever`.  No
            //     re-allocation, GC frees the rstr when the cell
            //     dies.
            //
            //   * pyre: `&str` has no stable backing-storage address
            //     by default, so this macro emits a fresh slot via
            //     `Box::leak` *every merge-point hit* — not once per
            //     JitCell.  The GreenKey HashMap content-de-dupes via
            //     `default_str_eq` / `default_str_hash`, so multiple
            //     slots with the same content collapse to a single
            //     cache entry, but the slot leaks themselves grow
            //     unboundedly with merge-point hit count for
            //     long-running programs.
            //
            // A structural fix (per-JitCell owned-string field
            // instead of leaked slot, e.g. reshaping
            // `GreenKey::values` from `Vec<i64>` to a typed enum
            // carrying `Box<str>` for str/unicode greens, with the
            // macro emitting a temporary the cache promotes on
            // insertion) is a larger refactor and is
            // intentionally deferred — a global intern side-table
            // was rejected as non-orthodox (RPython does not
            // maintain one).  Functional behavior matches RPython
            // (content-keyed compare/hash); only the lifetime /
            // allocation profile differs.
            Some(GreenTypeTag::Str) => quote! {
                (
                    majit_ir::make_str_slot(#spec_expr),
                    majit_ir::GreenType::Str,
                )
            },
            Some(GreenTypeTag::Unicode) => quote! {
                (
                    majit_ir::make_str_slot(#spec_expr),
                    majit_ir::GreenType::Unicode,
                )
            },
        }
    }

    /// `interp_jit.py:117-119` `can_enter_jit(next_instr=jumpto, pycode=...)`:
    /// the greens a back edge hands the driver name the interpreter state at the
    /// JUMP TARGET, not at the back-edge instruction.  `jumpto` is passed for
    /// `next_instr` — there is no second, current-position green.
    ///
    /// The `#[jit_interp]` marker form takes the target positionally and fills
    /// the greens from the declaration, so a declared position green (`pc`)
    /// would otherwise be read at the back edge.  Substituting the target for
    /// it is what makes `back_edge_structured`'s key equal the key the merge
    /// point at that target derives (`TraceCtx::merge_point_green_key_hash`) —
    /// which is the key `compile_loop` files the loop under
    /// (pyjitpl.py `original_boxes[:num_green_args]`).  Without it the
    /// loop is stored under a key nothing enters: the interpreter's
    /// `has_compiled_loop` misses forever, every back edge re-arms tracing, and
    /// the compiled artifact is never executed.
    fn subst_target_for_pc<'a>(expr: &'a Expr, pc: &Expr, target: &'a Expr) -> &'a Expr {
        if quote!(#expr).to_string() == quote!(#pc).to_string() {
            target
        } else {
            expr
        }
    }

    fn green_key_expr(
        target: &Expr,
        pc: &Expr,
        greens: &[Expr],
        green_type_tags: &[Option<green_type_tag::GreenTypeTag>],
    ) -> Option<TokenStream> {
        if greens.is_empty() {
            None
        } else {
            // Per-green dispatch through `majit_ir::GreenAsI64::__green_repr`
            // so the `(i64-bits, GreenType)` pair travels together for each
            // green expression. `with_types` builds the typed schema;
            // `warmstate.py comparekey _green_args_spec` keys per-type
            // `equal_whatever`/`hash_whatever` off the green's lltype, so a
            // Ref-typed green must compare by pointer identity, a Float by
            // bit pattern, and an Int by raw value — collapsing all to
            // `GreenType::Int` (the previous `GreenKey::new` shape) made
            // Float / Ref greens equal under bit-equal Int values they
            // should not be equal to.
            //
            // `green_type_tags` is the lockstep
            // `Vec<Option<GreenTypeTag>>` carried alongside `greens`
            // (`JitInterpConfig.green_type_tags`).  Tagged greens
            // bypass the trait dispatch with explicit casts so str /
            // unicode greens carry `GreenType::Str` (warmstate.py
            // ll_streq / ll_strhash routing).  Untagged greens
            // (`None`) keep the trait path unchanged.
            let green_reprs: Vec<TokenStream> = greens
                .iter()
                .enumerate()
                .map(|(i, expr)| {
                    let tag = green_type_tags.get(i).copied().flatten();
                    emit_green_repr(subst_target_for_pc(expr, pc, target), tag)
                })
                .collect();
            // Each green's `(i64, GreenType)` pair is bound to a local
            // ONCE — the green expressions must not be re-evaluated, so
            // the deferred key builder below closes over these `Copy`
            // locals rather than over the expressions.
            //
            // The slots ARE the declared greens (`pyjitpl.py`
            // `original_boxes[:num_green_args]` / `warmstate.py`
            // `JitCell.get_uhash`). A declared `pc` green is the target
            // (`subst_target_for_pc`); the target is not prepended in
            // front of that list.
            let slots: Vec<syn::Ident> = (0..green_reprs.len())
                .map(|i| {
                    syn::Ident::new(&format!("__green_slot{i}"), proc_macro2::Span::call_site())
                })
                .collect();
            let bind: Vec<TokenStream> = slots
                .iter()
                .zip(&green_reprs)
                .map(|(slot, repr)| quote! { let #slot = #repr; })
                .collect();
            // `get_uhash` unrolled over the declared greens — the count and
            // types are known here, exactly as upstream's
            // `green_args_name_spec` is fixed per JitCell class at
            // translation time (warmstate.py). No `values` /
            // `types` vectors are built to hash.
            let fold: Vec<TokenStream> = slots
                .iter()
                .map(|slot| {
                    quote! {
                        __green_hash =
                            majit_ir::green_uhash_step(__green_hash, #slot.1, #slot.0);
                    }
                })
                .collect();
            let values: Vec<TokenStream> = slots.iter().map(|slot| quote! { #slot.0 }).collect();
            let types: Vec<TokenStream> = slots.iter().map(|slot| quote! { #slot.1 }).collect();
            // `(hash, make_key)`: the hash is needed on every back edge, the
            // typed key only where a cell is installed.
            Some(quote! {
                {
                    #(#bind)*
                    let mut __green_hash: u64 = majit_ir::GREEN_UHASH_SEED;
                    #(#fold)*
                    (
                        __green_hash,
                        move || majit_ir::GreenKey::with_types(
                            ::std::vec![#(#values),*],
                            ::std::vec![#(#types),*],
                        ),
                    )
                }
            })
        }
    }

    struct MarkerRewriter {
        merge_fn_name: Ident,
        portal_green_args: Vec<Expr>,
        carried_green_args: Vec<Expr>,
        /// Whether this body's `jit_merge_point!` carries the `; state`
        /// single-executor close. `take_single_pass_finish` is set only by that
        /// machinery, so it is also what decides whether the back edge below
        /// has a terminal-return exit to emit.
        single_pass_close: bool,
        default_greens: Vec<Expr>,
        /// Whether the attribute spelled a `greens` key. `default_greens` being
        /// empty does not answer this: an omitted key and `greens = []` both
        /// arrive here as an empty `Vec`, and only one of them is a mistake.
        greens_declared: bool,
        default_green_type_tags: Vec<Option<green_type_tag::GreenTypeTag>>,
        recursive_entry: Option<Path>,
        finish_return: Option<FinishReturn>,
        /// Innermost scope last. A `let` is recorded after its statement, so
        /// a marker sees the bindings in scope at that point.
        scopes: Vec<Vec<ScopeBind>>,
    }

    impl VisitMut for MarkerRewriter {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // First recurse into children
            syn::visit_mut::visit_stmt_mut(self, stmt);

            // Check if this statement is a macro invocation
            if let syn::Stmt::Macro(stmt_macro) = stmt {
                let mac = &stmt_macro.mac;
                let path_str = mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");

                if path_str == "jit_merge_point" || path_str.ends_with("::jit_merge_point") {
                    let args =
                        syn::parse2::<MergePointArgs>(mac.tokens.clone()).unwrap_or_default();
                    let merge_fn = &self.merge_fn_name;
                    let driver = args.driver.unwrap_or_else(|| syn::parse_quote!(driver));
                    let env = args.env.unwrap_or_else(|| syn::parse_quote!(program));
                    let pc = args.pc.unwrap_or_else(|| syn::parse_quote!(pc));
                    let portal_green_args = &self.portal_green_args;
                    let carried_green_args = &self.carried_green_args;
                    let green_assigns = green_local_assigns(
                        &self.scopes,
                        &self.default_greens,
                        &self.default_green_type_tags,
                        &pc,
                    );
                    // jit_merge_point!() in #[jit_interp] dispatch portals
                    // expands to a single merge_wrapper invocation.  The wrapper
                    // (generate_merge_wrapper) clones the dispatch JitCode Arc and calls
                    // `driver.merge_point(...)` exactly once — there is no additional
                    // outer merge-point hook.  The BC_JIT_MERGE_POINT(_C) IR op lives
                    // inside the dispatch JitCode body (lower_dispatch_body),
                    // not in the outer Rust source.
                    //
                    // RPython parity: source-level jit_merge_point() is a codewriter
                    // marker; the runtime hook is the JitCode IR op
                    // (interp_jit.py).
                    //
                    // The `is_tracing()` guard here is a hot-path short-circuit
                    // (avoids the cold `__merge_*` call when not tracing).  It does NOT
                    // add a second merge-point dispatch — `driver.merge_point` guards
                    // again internally, but the closure runs only once.
                    let single_pass_finish_drain = self
                        .finish_return
                        .as_ref()
                        .map(|finish_return| finish_return.drain_single_pass(&driver))
                        .unwrap_or_default();
                    let new_tokens: TokenStream = if let Some(state) = &args.state {
                        // Single-pass close: after the walk, if the CloseLoop
                        // populated a walk-final (pc, reds) snapshot, transfer
                        // it into native state and resume the native loop at the
                        // close pc, skipping the body re-run. `#merge_fn` returns
                        // `None` whenever the walk did not populate reds.
                        quote! {
                            if #driver.is_tracing() {
                                let mut __mp_crn: ::core::option::Option<
                                    majit_metainterp::PortalResume
                                > = None;
                                let mut __mp_out = #merge_fn(
                                    &mut #driver,
                                    #env,
                                    #pc,
                                    &#state,
                                    #(#portal_green_args),*
                                    #(#carried_green_args),*
                                );
                                // pyjitpl.py run_blackhole_interp_to_cancel_tracing
                                // run_blackhole_interp_to_cancel_tracing: an
                                // abort can stop mid-source-opcode, where the
                                // walk's own pc names no resume position. The
                                // Abort arm staged the framestack; finish those
                                // opcodes in the blackhole here — the first
                                // point that also holds `state` — and take the
                                // resume from the merge point they reach.
                                // `None` when the walk did not abort or the
                                // conversion declined, leaving the source-pc
                                // handoff below untouched. The resume keeps
                                // `jitexc.py ContinueRunningNormally`'s
                                // green banks.
                                if let Some(__bh) = #driver.run_pending_abort_blackhole(
                                    &mut #state, #env,
                                ) {
                                    // `bhimpl_jit_merge_point` already built
                                    // the banks on the resume. Do not clone
                                    // them into the walk-close tuple; the
                                    // empty banks here are discarded when
                                    // `__mp_crn` is taken below.
                                    __mp_out = Some((
                                        __bh.resume_pc().unwrap_or(usize::MAX),
                                        ::std::vec::Vec::new(),
                                        majit_metainterp::ContinueRunningNormallyArgs::default(),
                                    ));
                                    __mp_crn = Some(__bh);
                                }
                                if let Some((__sp_pc, __sp_reds, __sp_args)) = __mp_out
                                {
                                    // Push the walk-mutated scalar state fields
                                    // (e.g. a walk-advanced `selected`) into
                                    // native `state`. The walk advances these on
                                    // the sym via BC_STORE_STATE_FIELD only;
                                    // native `state` is frozen at trace-start
                                    // (S_k), so without this the scalars stay
                                    // stale — and a `recover` that re-derives
                                    // caches FROM a scalar (stacksize from
                                    // selected) would then read the wrong index.
                                    // The values were captured off the still-live
                                    // sym inside the CloseLoop arm (the arm clears
                                    // the sym before returning, so they cannot be
                                    // re-read here) and stashed on the MetaInterp;
                                    // this consumes the stash. Must precede
                                    // `recover`, which refines storage-backed
                                    // caches from the current scalars. No-op with
                                    // no scalar state fields or when no CloseLoop
                                    // stashed values.
                                    #driver.writeback_scalar_state_fields(
                                        &mut #state,
                                    );
                                    // Ref scalars are stashed beside the ints
                                    // (`single_pass_ref_scalar_values`) and are
                                    // not part of the scalar vector. The bridge
                                    // resume hook writes them back; this close
                                    // owes the same transfer or native `state`
                                    // keeps the trace-start ref.
                                    #driver.writeback_ref_scalar_state_fields(
                                        &mut #state,
                                    );
                                    // Push the walk-final loop-carried virt-array
                                    // element values into native `state` too. A
                                    // field embedding a Rust `Vec` by value is
                                    // carved out of `synchronize_virtualizable`,
                                    // so its array stays frozen at trace-start
                                    // and this is its only writer; without it the
                                    // compiled-loop seed (extract_live reads
                                    // native state)
                                    // reflects the trace-start array and the loop
                                    // re-executes the peeled iteration, double-
                                    // firing any side-effecting residual. No-op
                                    // when the state has no virtualizable array.
                                    #driver.writeback_virt_array_state_fields(
                                        &mut #state,
                                    );
                                    // Transfer any loop-carried reds the walk
                                    // captured, then re-derive the
                                    // storage-backed cache fields (stacksize,
                                    // selected/storage refs) from the
                                    // walk-advanced shared storage via the
                                    // `recover` hook. `__sp_reds` is empty today
                                    // (merge-point red-operand slots don't map to
                                    // native state fields; the scalar write-back
                                    // above is the authoritative scalar source),
                                    // so this block is scaffolding for a future
                                    // JIT whose merge point carries real reds.
                                    if !__sp_reds.is_empty() {
                                        let __sp_meta = majit_metainterp::JitState::build_meta(
                                            &#state, __sp_pc, #env,
                                        );
                                        majit_metainterp::JitState::restore_values(
                                            &mut #state, &__sp_meta, &__sp_reds,
                                        );
                                    }
                                    majit_metainterp::JitState::recover_after_compiled_run(
                                        &mut #state,
                                    );
                                    #driver.log_single_pass_parity_resume(
                                        __sp_pc,
                                        &#state,
                                        #env,
                                    );
                                    #driver.arm_single_pass_label_entry_on_next_back_edge(
                                        &#state,
                                    );
                                    // RPython pyjitpl.py raise_if_successful ->
                                    // 3072-3091 parity: a successful translated
                                    // CloseLoop raises ContinueRunningNormally,
                                    // returning to the interpreter. The compiled
                                    // loop is entered later via can_enter_jit /
                                    // warmstate, not by immediate direct entry.
                                    #driver.discard_single_pass_resume();
                                    // `warmspot.py ll_portal_runner` re-enters
                                    // the portal with every green from
                                    // `jitexc.py ContinueRunningNormally`.
                                    // An abort blackhole already built those
                                    // banks; a single-pass close carries them
                                    // as `TraceCtx::portal_resume_args`.
                                    let __crn_resume = match __mp_crn {
                                        Some(__resume) => __resume,
                                        None => {
                                            majit_metainterp::PortalResume::ContinueRunningNormally(
                                                __sp_args,
                                            )
                                        }
                                    };
                                    let __resume_pc = __crn_resume.resume_pc();
                                    if let Some(__crn_args) = __crn_resume.args() {
                                        #green_assigns
                                    }
                                    #driver.recycle_portal_resume(__crn_resume);
                                    // A terminal dispatch return (Finish) means
                                    // the interpreted function has returned:
                                    // exit the native dispatch loop and run its
                                    // own post-loop return exactly once. Resuming
                                    // at the captured pc instead would re-enter
                                    // the loop body when the terminal opcode is
                                    // mid-program (its post-advance pc lands
                                    // before the program end). A CloseLoop
                                    // back-edge keeps interpreting from the
                                    // merge-point pc.
                                    // The drain belongs INSIDE the latch it
                                    // reads.  Ahead of it, it fires on every
                                    // single-pass close -- an ordinary
                                    // `CloseLoop` back edge included -- and a
                                    // stale value latch would return from the
                                    // portal where the loop must keep running.
                                    if #driver.take_single_pass_finish() {
                                        #single_pass_finish_drain
                                        break;
                                    }
                                    // Greens are assigned first so that
                                    // epilogue reads the banks.
                                    let Some(__resume_pc) = __resume_pc else {
                                        break;
                                    };
                                    #pc = __resume_pc;
                                    continue;
                                }
                            }
                        }
                    } else {
                        quote! {
                            if #driver.is_tracing() {
                                #merge_fn(
                                    &mut #driver,
                                    #env,
                                    #pc,
                                    &state,
                                    #(#portal_green_args),*
                                    #(#carried_green_args),*
                                );
                            }
                        }
                    };
                    // `; state` selects the single-executor close: the native
                    // loop resumes at the green pc the merge point reports.  A
                    // driver with no declared greens reports none, and the
                    // failure is silent in release —
                    // `run_pending_abort_blackhole` reaches
                    // `ContinueRunningNormally` with an empty `green_int`
                    // (`jitdriver.rs`), sets `single_pass_finish`, and
                    // the dispatch loop ends after exactly `threshold` passes,
                    // returning a truncated result that still looks plausible.
                    // The `debug_assert!` naming that contract is compiled out
                    // in release, so only the ungated `eprintln!` beside it ever
                    // fires.  Refuse at compile time rather than truncate at
                    // run time.
                    //
                    // Wrapped in a block rather than replacing the expansion:
                    // `parse2` below wants exactly one `Stmt`, and keeping the
                    // original tokens means the error reported is this one
                    // instead of a cascade of unresolved names.
                    // Only a missing `greens` key is refused. An explicit
                    // `greens = []` is a supported spelling — the fixtures that
                    // grade the empty-greens encoding use it — and compiles.
                    let new_tokens = if !self.greens_declared {
                        let driver_name = quote!(#driver).to_string();
                        // State the required metadata rather than enumerating
                        // merge-point spellings, which may grow independently.
                        let msg = format!(
                            "`jit_merge_point!` on `{driver_name}` requires a `greens` key \
                             on the enclosing `#[jit_interp]` attribute, and this attribute \
                             has none. A merge point resumes the interpreter at the green \
                             pc it reports; with no greens declared it reports none, so \
                             the tracing walk never advances and what the JIT ships is \
                             degenerate — a compiled loop that runs zero iterations, or \
                             nothing at all. Declare the greens the loop is keyed on, e.g. \
                             `greens = [pc, program]`. If the empty set is deliberate — a \
                             fixture grading the empty-greens encoding — spell it \
                             `greens = []`, which is supported and is not refused here."
                        );
                        quote! { { compile_error!(#msg); #new_tokens } }
                    } else {
                        new_tokens
                    };
                    *stmt =
                        syn::parse2(new_tokens).expect("failed to parse merge_point replacement");
                }
            }
        }

        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            // First recurse
            syn::visit_mut::visit_expr_mut(self, expr);

            // `recursive_portal_call!(driver, green0, green1, ...)` — the
            // concrete (non-JIT) path re-enters the portal via the declared
            // `recursive_entry` function, forwarding the greens positionally
            // (jitdriver declaration order).  The dispatch JitCode lowers the
            // same intrinsic to `BC_RECURSIVE_CALL_*`.
            if let Expr::Macro(em) = expr {
                let path_str = em
                    .mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if path_str == "recursive_portal_call"
                    || path_str.ends_with("::recursive_portal_call")
                {
                    let args = em
                        .mac
                        .parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
                        .expect("recursive_portal_call! takes `driver, green0, green1, ...`");
                    let mut iter = args.into_iter();
                    let _driver = iter
                        .next()
                        .expect("recursive_portal_call! requires a driver argument");
                    let greens: Vec<Expr> = iter.collect();
                    let entry = self.recursive_entry.as_ref().unwrap_or_else(|| {
                        panic!(
                            "recursive_portal_call! used but `#[jit_interp(..)]` declares no \
                             `recursive_entry = <fn path>` for the concrete fallback"
                        )
                    });
                    let new_tokens = quote! { #entry(#(#greens),*) };
                    *expr = syn::parse2(new_tokens)
                        .expect("failed to parse recursive_portal_call concrete fallback");
                }
            }
        }

        fn visit_block_mut(&mut self, block: &mut syn::Block) {
            self.scopes.push(Vec::new());
            let mut new_stmts = Vec::new();
            let mut i = 0;
            while i < block.stmts.len() {
                let stmt = &block.stmts[i];

                // Check if this is can_enter_jit!(driver, target, state, env, pre_run, ...)
                if let syn::Stmt::Macro(stmt_macro) = stmt {
                    let mac = &stmt_macro.mac;
                    let path_str = mac
                        .path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");

                    if path_str == "can_enter_jit" || path_str.ends_with("::can_enter_jit") {
                        let tokens = &mac.tokens;
                        if let Ok(args) = syn::parse2::<CanEnterJitArgs>(tokens.clone()) {
                            let driver_expr = &args.driver;
                            let target_expr = &args.target;
                            let state_expr = &args.state;
                            let env_expr = &args.env;
                            let pre_run_expr = &args.pre_run;
                            let pc_expr = args
                                .pc
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(pc));
                            let _stacksize_expr = args
                                .stacksize
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(stacksize));
                            let (greens, green_type_tags) = if args.greens.is_empty() {
                                (
                                    self.default_greens.clone(),
                                    self.default_green_type_tags.clone(),
                                )
                            } else {
                                // RPython `can_enter_jit` and
                                // `jit_merge_point` reference the same
                                // `JitDriver` greens spec — the marker op
                                // args sit in fixed positional slots whose
                                // lltype is fixed by declaration
                                // (`warmstate.py:564 _green_args_spec`,
                                // `support.py decode_hp_hint_args`
                                // asserts on count mismatch at translation
                                // time).  Positional inheritance of
                                // declaration tags therefore matches
                                // RPython parity: a `str`-tagged green
                                // stays Str-keyed through the override
                                // path, routing through the canonical
                                // slot ABI rather than silently falling
                                // to the blanket `GreenAsI64 for &T`
                                // (Ref / pointer identity) implementation.
                                //
                                // If a downstream override expression has
                                // a different Rust type than the
                                // declaration's tag indicates (e.g. tag
                                // `Int` with `&str` override), the
                                // emitted explicit cast (`(<expr>) as
                                // i64`) fails at compile time — fail-loud
                                // at the macro / type-check boundary
                                // rather than at runtime with a
                                // misshaped key.
                                //
                                // Marker arity is structurally fixed in
                                // RPython, so a count mismatch is a
                                // hard user error. pyre fails loud at
                                // proc-macro expansion (compile-time)
                                // with a clear message rather than
                                // silently falling back to an untagged
                                // trait path that would emit a misshaped
                                // key schema.
                                if args.greens.len() != self.default_green_type_tags.len() {
                                    panic!(
                                        "can_enter_jit! override green count {} does not match \
                                         the JitDriver declaration's green count {} — RPython \
                                         marker arity is fixed (rpython/jit/codewriter/support.py \
                                         decode_hp_hint_args asserts on mismatch). Fix the \
                                         override expression list to match the declaration's \
                                         `greens=[...]` arity.",
                                        args.greens.len(),
                                        self.default_green_type_tags.len(),
                                    );
                                }
                                (args.greens.clone(), self.default_green_type_tags.clone())
                            };
                            // compile.py:711 parity: back_edge returns
                            // Some(resume_pc) on a guard failure (blackhole
                            // resume) or a back-edge JUMP.
                            // state.restore_values already restores all
                            // state fields (including stacksize) from the
                            // compiled loop's exit state, so no explicit
                            // stacksize reset is needed here.
                            //
                            // A run that ended in FINISH is NOT a resume: the
                            // traced function returned, so the driver's FINISH
                            // latch is drained first and turned into this
                            // portal's own `return`. The resume_pc offered
                            // alongside it is the back edge, and taking it
                            // would re-run the loop the compiled run already
                            // completed — once per remaining iteration.
                            let green_assigns = green_local_assigns(
                                &self.scopes,
                                &greens,
                                &green_type_tags,
                                &pc_expr,
                            );
                            let finish_drain: TokenStream = self
                                .finish_return
                                .as_ref()
                                .map(|finish_return| finish_return.drain(driver_expr))
                                .unwrap_or_default();
                            let call: TokenStream = if let Some(green_key) =
                                green_key_expr(target_expr, &pc_expr, &greens, &green_type_tags)
                            {
                                quote! {
                                    {
                                        let (__green_hash, __green_key) = #green_key;
                                        #driver_expr.back_edge_structured(__green_hash, __green_key, #target_expr, #state_expr, #env_expr, #pre_run_expr)
                                    }
                                }
                            } else {
                                quote! {
                                    #driver_expr.back_edge(#target_expr, #state_expr, #env_expr, #pre_run_expr)
                                }
                            };
                            // The back edge's spelling of the
                            // `jit_merge_point!` hook's terminal-return exit. A
                            // guard that bridges enters the walk at the guard's
                            // own position, so the walk reaches the interpreted
                            // function's return here just as it can at a merge
                            // point; the position reported alongside names
                            // nothing to resume at, and assigning it to
                            // `#pc_expr` decodes an out-of-range index in any
                            // dispatch loop not bottom-tested on its program
                            // length.
                            //
                            // Emitted only for a body whose merge point carries
                            // `; state`, because only that machinery ever sets
                            // the flag. A bare `jit_merge_point!()` body would
                            // get a `break` that can never be taken, and a bare
                            // `break` gives an expression-position dispatch
                            // `loop` the type `()` — so an unconditional
                            // emission stops such a body from compiling at all.
                            let single_pass_finish_exit: TokenStream = if self.single_pass_close {
                                // Same latch, same drain as the merge-point
                                // close: the dispatch JitCode has already run
                                // the source post-loop epilogue, so breaking
                                // without draining runs that suffix a second
                                // time natively.  With no value published the
                                // drain falls through and the `break` stands.
                                let drain = self
                                    .finish_return
                                    .as_ref()
                                    .map(|finish_return| {
                                        finish_return.drain_single_pass(driver_expr)
                                    })
                                    .unwrap_or_default();
                                quote! {
                                    if #driver_expr.take_single_pass_finish() {
                                        #drain
                                        break;
                                    }
                                }
                            } else {
                                TokenStream::new()
                            };
                            let back_edge: TokenStream = quote! {
                                {
                                    let __back_edge_resume = #call;
                                    let __had_resume = __back_edge_resume.is_some();
                                    let __resume_pc = __back_edge_resume
                                        .as_ref()
                                        .and_then(|__crn_resume| __crn_resume.resume_pc());
                                    if let Some(ref __crn_resume) = __back_edge_resume {
                                        // `warmspot.py ll_portal_runner` writes
                                        // every `jitexc.py ContinueRunningNormally`
                                        // green back before the loop decides
                                        // whether the frame is finished.
                                        if let Some(__crn_args) = __crn_resume.args() {
                                            #green_assigns
                                        }
                                    }
                                    if let Some(__crn_resume) = __back_edge_resume {
                                        #driver_expr.recycle_portal_resume(__crn_resume);
                                    }
                                    #finish_drain
                                    if __had_resume {
                                        #single_pass_finish_exit
                                        // Same sentinel as the merge-point close.
                                        // A compiled guard that ran the frame to
                                        // completion reports no bytecode pc;
                                        // storing it makes the next dispatch
                                        // fail. Leaving the loop returns the
                                        // status already written on `state`.
                                        let Some(__resume_pc) = __resume_pc else {
                                            break;
                                        };
                                        #pc_expr = __resume_pc;
                                        continue;
                                    }
                                }
                            };
                            let parsed: syn::Stmt =
                                syn::parse2(back_edge).expect("failed to parse back_edge");
                            new_stmts.push(parsed);
                            i += 1;
                            continue;
                        }
                    }
                }

                let mut cloned = block.stmts[i].clone();
                self.visit_stmt_mut(&mut cloned);
                if let syn::Stmt::Local(local) = &block.stmts[i]
                    && let Some(bind) = scope_bind_from_local(local)
                {
                    self.scopes
                        .last_mut()
                        .expect("visit_block_mut pushed a scope")
                        .push(bind);
                }
                new_stmts.push(cloned);
                i += 1;
            }
            block.stmts = new_stmts;
            self.scopes.pop();
        }
    }

    let mut cloned_block = block.clone();
    // Whether any `jit_merge_point!` in this body carries `; state`. Read
    // before the rewrite because the back edge and the merge point are separate
    // statements visited in source order, and a dispatch loop reaches its back
    // edge inside an opcode arm that can precede the merge point in the tree.
    let scan = scan_single_pass_close(&cloned_block);

    let mut rewriter = MarkerRewriter {
        merge_fn_name: merge_fn_name.clone(),
        portal_green_args: portal_green_args.to_vec(),
        carried_green_args: carried_green_args.to_vec(),
        single_pass_close: scan.found,
        default_greens: default_greens.to_vec(),
        greens_declared,
        default_green_type_tags: default_green_type_tags.to_vec(),
        recursive_entry: recursive_entry.cloned(),
        finish_return: finish_return.cloned(),
        scopes: vec![param_scope_binds(func_sig)],
    };
    let traced_loop_at = traced_loop_stmt_index(&cloned_block.stmts);
    rewriter.visit_block_mut(&mut cloned_block);
    rewriter.scopes.pop();

    // warmspot.py ll_portal_runner: maybe enter from the function's start,
    // before the interpreter loop. Only the `; state` form has a live state
    // handle to extract reds from, which is what compiled entry and
    // force_start_tracing both need.
    let mut runner_door = None;
    let entry_door = if let Some(args) = scan.first_state {
        let driver = args
            .driver
            .clone()
            .unwrap_or_else(|| syn::parse_quote!(driver));
        let env = args
            .env
            .clone()
            .unwrap_or_else(|| syn::parse_quote!(program));
        let pc = args.pc.clone().unwrap_or_else(|| syn::parse_quote!(pc));
        let state = args.state.clone().expect("scan only stores state merges");
        let finish_drain = finish_return
            .map(|finish_return| finish_return.drain(&driver))
            .unwrap_or_default();
        // Same emission as the back edge: `; state` is what sets the latch.
        // With no finish projection the body is empty, and the take still
        // clears the flag so a later merge point does not observe it.
        let single_pass_drain = finish_return
            .map(|finish_return| finish_return.drain_single_pass(&driver))
            .unwrap_or_default();
        match green_key_expr(&pc, &pc, default_greens, default_green_type_tags) {
            // The block is the condition of `if #door { loop }`. A finished
            // frame reports `usize::MAX` and has no bytecode pc; assigning
            // that and entering the loop dispatches off the end of the
            // program. Skipping the loop leaves the status the run already
            // stored for the function's own epilogue. A real resume pc still
            // enters the loop there.
            Some(key) => {
                runner_door = Some(RunnerDoor {
                    key: key.clone(),
                    driver: driver.clone(),
                    pc: pc.clone(),
                    state: state.clone(),
                    env: env.clone(),
                });
                if insert_entry_door {
                    quote! {
                        {
                            let (__green_hash, __make_key) = #key;
                            let __run_dispatch = match #driver.function_entry_structured(
                                __green_hash,
                                __make_key,
                                #pc,
                                &mut #state,
                                #env,
                            ) {
                                Some(__resume) if __resume == usize::MAX => false,
                                Some(__resume) => {
                                    #pc = __resume;
                                    true
                                }
                                None => true,
                            };
                            #finish_drain
                            // `warmspot.py ll_portal_runner` returns
                            // `DoneWithThisFrameRef.result` from
                            // `execute_assembler`, `blackhole.py
                            // _done_with_this_frame`, or `pyjitpl.py
                            // MetaInterp.compile_done_with_this_frame`. The
                            // bridge walk publishes that result on the
                            // single-pass latch and reports `usize::MAX`.
                            if #driver.take_single_pass_finish() {
                                #single_pass_drain
                            }
                            __run_dispatch
                        }
                    }
                } else {
                    quote! {}
                }
            }
            None => quote! {},
        }
    } else {
        quote! {}
    };

    let stmts = insert_before_traced_loop(cloned_block.stmts, entry_door, traced_loop_at);
    (stmts, runner_door)
}

/// `ll_portal_runner` sits after driver/pc/state exist and immediately
/// before the loop that contains `jit_merge_point`, not a preceding
/// setup loop.
fn insert_before_traced_loop(
    stmts: Vec<syn::Stmt>,
    door: TokenStream,
    traced_loop_at: Option<usize>,
) -> Vec<syn::Stmt> {
    if door.is_empty() {
        return stmts;
    }
    let Some(at) = traced_loop_at else {
        let door_stmt: syn::Stmt =
            syn::parse2(door).expect("function-entry door must parse as a statement");
        let mut out = stmts;
        out.insert(0, door_stmt);
        return out;
    };
    let at = at.min(stmts.len());
    if at == stmts.len() {
        let door_stmt: syn::Stmt =
            syn::parse2(door).expect("function-entry door must parse as a statement");
        let mut out = stmts;
        out.push(door_stmt);
        return out;
    }
    // `door` is a bool: true enters the dispatch loop, false skips it and
    // falls into the function's own epilogue. One check per call, not per
    // opcode.
    let syn::Stmt::Expr(loop_expr, _) = &stmts[at] else {
        let door_stmt: syn::Stmt =
            syn::parse2(door).expect("function-entry door must parse as a statement");
        let mut out = stmts;
        out.insert(at, door_stmt);
        return out;
    };
    let wrapped: syn::Stmt = syn::parse2(quote! {
        if #door {
            #loop_expr;
        }
    })
    .expect("function-entry door must wrap the dispatch loop");
    let mut out = stmts;
    out[at] = wrapped;
    out
}

fn traced_loop_stmt_index(stmts: &[syn::Stmt]) -> Option<usize> {
    stmts.iter().position(stmt_is_traced_loop)
}

fn stmt_is_traced_loop(stmt: &syn::Stmt) -> bool {
    let syn::Stmt::Expr(expr, _) = stmt else {
        return false;
    };
    let body = match expr {
        syn::Expr::Loop(l) => &l.body,
        syn::Expr::While(w) => &w.body,
        _ => return false,
    };
    body.stmts.iter().any(|s| match s {
        syn::Stmt::Macro(m) => m
            .mac
            .path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "jit_merge_point"),
        syn::Stmt::Expr(syn::Expr::Macro(m), _) => m
            .mac
            .path
            .segments
            .last()
            .is_some_and(|seg| seg.ident == "jit_merge_point"),
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    /// A body that spells both merge-point forms is rejected, and the
    /// diagnostic names the field that cannot hold both answers.
    #[test]
    fn a_body_mixing_merge_point_forms_is_rejected() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                loop {
                    jit_merge_point!(driver, program, pc; state);
                    if done {
                        break;
                    }
                    jit_merge_point!();
                }
                0
            }
        };
        let err = validate_merge_point_forms(&func)
            .expect_err("one function may not declare two ownership modes");
        // Both halves are reported, one span each, so the author sees the two
        // markers that disagree rather than only the first.
        let messages: Vec<String> = err.into_iter().map(|e| e.to_string()).collect();
        assert_eq!(
            messages.len(),
            2,
            "both markers must be named: {messages:?}"
        );
        let message = messages.join(" ");
        assert!(
            message.contains("outer_executor_owns_state"),
            "the diagnostic must name the field that cannot hold both \
             answers: {message}"
        );
    }

    /// Repeating ONE form is not a mix: a body is free to spell the marker as
    /// often as it needs to.
    #[test]
    fn a_body_repeating_one_merge_point_form_is_accepted() {
        let single_pass: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                loop {
                    jit_merge_point!(driver, program, pc; state);
                    jit_merge_point!(driver, program, pc; state);
                }
            }
        };
        assert!(validate_merge_point_forms(&single_pass).is_ok());

        let outer_owned: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                loop {
                    jit_merge_point!();
                    jit_merge_point!();
                }
            }
        };
        assert!(validate_merge_point_forms(&outer_owned).is_ok());
    }

    /// A path in two vocabularies is rejected at parse time, naming the route
    /// that wins. Without the check the `calls` policy is simply never
    /// consulted and nothing downstream can tell the author.
    #[test]
    fn a_path_in_two_call_vocabularies_is_rejected() {
        let calls = vec![CallEntry {
            path: parse_quote!(unwrap_word),
            policy: Some(CallPolicyKind::ElidableIntCannotRaise),
        }];
        let native_identity: Vec<Path> = vec![parse_quote!(unwrap_word)];

        let err = reject_overlapping_call_vocabularies(&calls, &[], &[], &native_identity)
            .expect_err("a path in both `native_identity` and `calls` must not parse");
        let message = err.to_string();
        assert!(
            message.contains("`unwrap_word`")
                && message.contains("`native_identity`")
                && message.contains("`calls`"),
            "the diagnostic must name the path and both vocabularies: {message}"
        );
    }

    /// Two entries under one key stay legal: `calls` is filled from `calls`
    /// and `helpers` both, and naming a discovered helper there is how it
    /// gets its policy.
    #[test]
    fn a_path_declared_twice_in_one_vocabulary_is_accepted() {
        let calls = vec![
            CallEntry {
                path: parse_quote!(helper),
                policy: None,
            },
            CallEntry {
                path: parse_quote!(helper),
                policy: Some(CallPolicyKind::ResidualInt),
            },
        ];
        reject_overlapping_call_vocabularies(&calls, &[], &[], &[])
            .expect("a repeated `calls` entry is how `helpers` gets its policy");
    }

    #[test]
    fn an_unambiguous_imported_leaf_resolves_its_inlined_base() {
        let mut prefixes = std::collections::HashMap::new();
        prefixes.insert(
            vec!["crate".into(), "storage".into(), "Queue".into()],
            ("base".into(), parse_quote!(crate::storage::ListBase)),
        );
        let queue: Path = parse_quote!(Queue);
        let owner = declaring_struct_of(
            &prefixes,
            |path, field| {
                path.segments.last().is_some_and(|segment| {
                    segment.ident == "ListBase" && matches!(field, "head" | "size")
                })
            },
            &queue,
            "head",
        );
        assert_eq!(
            owner
                .segments
                .iter()
                .map(|segment| segment.ident.to_string())
                .collect::<Vec<_>>(),
            ["crate", "storage", "ListBase"],
        );
    }

    #[test]
    fn an_ambiguous_imported_leaf_does_not_choose_an_inlined_base() {
        let prefixes = std::collections::HashMap::from([
            (
                vec!["left".into(), "Queue".into()],
                ("base".into(), parse_quote!(left::ListBase)),
            ),
            (
                vec!["right".into(), "Queue".into()],
                ("base".into(), parse_quote!(right::ListBase)),
            ),
        ]);
        assert!(
            inlined_prefix_entry(&prefixes, &["Queue".into()]).is_none(),
            "a bare leaf shared by two configured types must fail closed",
        );
    }

    #[test]
    fn parse_helpers_list_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [helper_add, helper_sub, helper_mul]
        };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].path.segments.last().unwrap().ident.to_string(),
            "helper_add"
        );
        assert!(result[0].policy.is_none());
        assert_eq!(
            result[1].path.segments.last().unwrap().ident.to_string(),
            "helper_sub"
        );
        assert_eq!(
            result[2].path.segments.last().unwrap().ident.to_string(),
            "helper_mul"
        );
    }

    #[test]
    fn parse_helpers_list_empty() {
        let tokens: proc_macro2::TokenStream = parse_quote! { [] };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert!(result.is_empty());
    }

    #[test]
    fn parse_helpers_list_with_path() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [module::helper_a, helper_b]
        };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        // First has two path segments
        assert_eq!(result[0].path.segments.len(), 2);
        assert_eq!(result[0].path.segments[0].ident.to_string(), "module");
        assert_eq!(result[0].path.segments[1].ident.to_string(), "helper_a");
    }

    /// Wrapper to make `parse_helpers_list` testable via `syn::parse2`.
    struct HelpersListWrapper(Vec<CallEntry>);
    impl Parse for HelpersListWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_helpers_list(input)?))
        }
    }

    /// Wrapper to make `parse_call_map` testable via `syn::parse2`.
    struct CallMapWrapper(Vec<CallEntry>);
    impl std::fmt::Debug for CallMapWrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_list()
                .entries(self.0.iter().map(|e| {
                    e.path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default()
                }))
                .finish()
        }
    }
    impl Parse for CallMapWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_call_map(input)?))
        }
    }

    #[test]
    fn parse_call_map_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { foo, bar => may_force_int }
        };
        let result: Vec<CallEntry> = syn::parse2::<CallMapWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].path.segments.last().unwrap().ident.to_string(),
            "foo"
        );
        assert!(result[0].policy.is_none());
        assert_eq!(
            result[1].path.segments.last().unwrap().ident.to_string(),
            "bar"
        );
        assert_eq!(result[1].policy, Some(CallPolicyKind::MayForceInt));
    }

    #[test]
    fn parse_call_map_rejects_slot_effect_blocks() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { pop_only { reads: [stackpos] } }
        };
        let err = syn::parse2::<CallMapWrapper>(tokens).unwrap_err();
        assert!(
            err.to_string().contains("expected identifier")
                || err.to_string().contains("expected `,`"),
            "expected call-map parser to reject slot-effect block, got: {err}",
        );
    }

    struct VirtualizableWrapper(VirtualizableDecl);
    impl Parse for VirtualizableWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_virtualizable_decl(input)?))
        }
    }

    struct StateFieldsWrapper(StateFieldsConfig);
    impl Parse for StateFieldsWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_state_fields(input)?))
        }
    }

    #[test]
    fn parse_state_fields_accepts_opaque_pass_through() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                storage: opaque(test_fixtures::storage::Storage),
                selected: int,
                tape: [int; virt],
            }
        };
        let parsed = syn::parse2::<StateFieldsWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.fields.len(), 3);
        assert_eq!(parsed.fields[0].name.to_string(), "storage");
        match &parsed.fields[0].kind {
            StateFieldKind::Opaque(p) => {
                assert_eq!(p.segments.last().unwrap().ident.to_string(), "Storage");
            }
            _ => panic!("expected Opaque variant"),
        }
        assert!(matches!(
            parsed.fields[1].kind,
            StateFieldKind::Scalar { .. }
        ));
        assert!(matches!(
            parsed.fields[2].kind,
            StateFieldKind::VirtArray(_)
        ));
    }

    #[test]
    fn parse_state_fields_accepts_str_red() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                pos: int,
                s: str,
            }
        };
        let parsed = syn::parse2::<StateFieldsWrapper>(tokens).unwrap().0;
        assert!(matches!(
            parsed.fields[0].kind,
            StateFieldKind::Scalar { .. }
        ));
        assert!(matches!(parsed.fields[1].kind, StateFieldKind::Str));
    }

    #[test]
    fn parse_state_fields_accepts_ref_scalar() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                a: int,
                sel: ref(crate::Stack),
            }
        };
        let parsed = syn::parse2::<StateFieldsWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.fields.len(), 2);
        assert!(matches!(
            parsed.fields[0].kind,
            StateFieldKind::Scalar { .. }
        ));
        match &parsed.fields[1].kind {
            StateFieldKind::Ref(p) => {
                assert_eq!(p.segments.last().unwrap().ident.to_string(), "Stack");
            }
            _ => panic!("expected Ref variant"),
        }
    }

    #[test]
    fn parse_state_fields_rejects_unknown_scalar_type() {
        // `float` is a supported scalar type now; use a genuinely unsupported
        // one (`u32` is a real type but not a valid state-field IR type).
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { val: u32 }
        };
        let err = match syn::parse2::<StateFieldsWrapper>(tokens) {
            Ok(_) => panic!("expected parse error for unknown scalar type"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unsupported type `u32`")
                && msg.contains("supported: `int`, `int(<TypePath>)`, `float`"),
            "msg: {msg}"
        );
    }

    #[test]
    fn parse_virtualizable_decl_keeps_direct_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: { next_instr: int @ NEXT_INSTR_OFFSET },
                arrays: { locals_w: ref @ LOCALS_OFFSET },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Direct {
                field_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(quote::quote!(#field_offset).to_string(), "LOCALS_OFFSET");
                assert!(length_offset.is_none());
                assert!(items_offset.is_none());
            }
            VableArrayLayoutDecl::Embedded { .. } => {
                panic!("expected direct array layout");
            }
        }
    }

    #[test]
    fn parse_virtualizable_decl_supports_embedded_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: {},
                arrays: {
                    stack: int @ (STORAGEPOOL_DATA_PTRS_OFFSET + SLOT) {
                        ptr_offset: 0,
                        length_offset: STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET,
                        items_offset: 0,
                    },
                },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Embedded {
                field_offset,
                ptr_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(
                    quote::quote!(#field_offset).to_string(),
                    "STORAGEPOOL_DATA_PTRS_OFFSET + SLOT"
                );
                assert_eq!(quote::quote!(#ptr_offset).to_string(), "0");
                assert_eq!(
                    quote::quote!(#length_offset).to_string(),
                    "STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET"
                );
                assert_eq!(quote::quote!(#items_offset).to_string(), "0");
            }
            VableArrayLayoutDecl::Direct { .. } => {
                panic!("expected embedded array layout");
            }
        }
    }

    fn validate(config_tokens: proc_macro2::TokenStream, func: ItemFn) -> syn::Result<()> {
        let config: JitInterpConfig = syn::parse2(config_tokens).unwrap();
        validate_state_fields(&config, &func)
    }

    /// A plain `[int]` array stored inside the traced loop must be rejected:
    /// the loop-carried element does not read back its heap value.
    #[test]
    fn validate_rejects_loop_carried_plain_array() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut state = S { regs: vec![0; 3] };
                let mut pc: usize = 0;
                loop {
                    jit_merge_point!();
                    let op = program[pc];
                    match op {
                        0 => { state.regs[0] = state.regs[0] + 1; pc += 1; }
                        _ => return state.regs[0],
                    }
                }
            }
        };
        let err = validate(
            quote! { state = S, env = Bytecode, greens = [pc, program],
            state_fields = { regs: [int] } },
            func,
        )
        .expect_err("loop-carried plain [int] store must be rejected");
        assert!(err.to_string().contains("[int; virt]"));
        assert!(err.to_string().contains("regs"));
    }

    /// `[int; virt]` is the supported loop-carried form — accepted.
    #[test]
    fn validate_accepts_virt_array() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut state = S { regs: vec![0; 3] };
                let mut pc: usize = 0;
                loop {
                    jit_merge_point!();
                    let op = program[pc];
                    match op {
                        0 => { state.regs[0] = state.regs[0] + 1; pc += 1; }
                        _ => return state.regs[0],
                    }
                }
            }
        };
        validate(
            quote! { state = S, env = Bytecode, greens = [pc, program],
            state_fields = { regs: [int; virt] } },
            func,
        )
        .expect("virt array must be accepted");
    }

    /// A plain `[int]` array only *read* in the loop is loop-invariant and
    /// stays valid — no false positive.
    #[test]
    fn validate_accepts_read_only_plain_array() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut state = S { regs: vec![0; 3] };
                let mut pc: usize = 0;
                let mut acc: i64 = 0;
                loop {
                    jit_merge_point!();
                    let op = program[pc];
                    match op {
                        0 => { acc += state.regs[0]; pc += 1; }
                        _ => return acc,
                    }
                }
            }
        };
        validate(
            quote! { state = S, env = Bytecode, greens = [pc, program],
            state_fields = { regs: [int] } },
            func,
        )
        .expect("read-only plain [int] array must be accepted");
    }

    /// An unsupported `state_fields` entry is reported the way it was
    /// declared, brackets included.
    ///
    /// `parse_state_fields` accepts `[<ident>]` for any element type and
    /// `codegen_state` is where the int-only restriction on plain arrays
    /// lives, so `[float]` is refused there. Rendering the element type alone
    /// printed `regs: float` — the scalar spelling the same message lists as
    /// supported — so the reader was told their declaration was both
    /// supported and not.
    #[test]
    fn an_unsupported_state_field_is_reported_as_declared() {
        fn expand(fields: proc_macro2::TokenStream) -> String {
            let config: JitInterpConfig = syn::parse2(quote! {
                state = S,
                env = Bytecode,
                greens = [],
                state_fields = { #fields },
            })
            .expect("fixture attribute must parse");
            let func: ItemFn = parse_quote! {
                fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                    let mut driver: majit_metainterp::JitDriver<S> =
                        majit_metainterp::JitDriver::new(threshold);
                    let mut pc: usize = 0;
                    let mut state = S { acc: 0 };
                    while pc < program.len() {
                        jit_merge_point!();
                        let op = program[pc];
                        pc += 1;
                        match op {
                            0 => { state.acc += 1; }
                            _ => break,
                        }
                    }
                    state.acc
                }
            };
            transform_jit_interp(config, func).to_string()
        }

        let flat_float = expand(quote! { acc: int, regs: [float] });
        assert!(
            flat_float.contains("compile_error"),
            "a plain `[float]` array has no lowering and must be refused:\n\
             {flat_float}"
        );
        assert!(
            flat_float.contains("regs: [float]"),
            "the refusal must render the declaration, not the element type — \
             `regs: float` names the supported scalar form instead of the \
             rejected array one:\n{flat_float}"
        );

        // The control: the same array over `int` expands cleanly, so the
        // assertions above are about the element type and not about the
        // fixture failing for some other reason.
        let flat_int = expand(quote! { acc: int, regs: [int] });
        assert!(
            !flat_int.contains("compile_error"),
            "`[int]` is the supported plain array:\n{flat_int}"
        );
    }

    /// A merge point whose enclosing attribute omits the `greens` key must be
    /// refused at expansion time.
    ///
    /// The second arm is the whole point of the test, not a bonus assertion.
    /// The two expansions differ by exactly one attribute entry, so a fixture
    /// that stops expanding for some unrelated reason emits `compile_error!`
    /// for *both* — and only the `with_greens` assertion notices. Checked on
    /// its own, the first assertion passes just as happily on a typo'd
    /// attribute as on a working guard, which would make this an oracle that
    /// cannot fail.
    ///
    /// THE THIRD ARM IS NOT A THIRD COPY OF THE SECOND. `greens = []` and
    /// `greens = [pc, program]` are both "not refused", but they are not
    /// refused for different reasons: the second declares greens, the third
    /// declares that there are none. The refusal reads the KEY, not the value,
    /// and the third arm is the only thing in this crate that says so — drop
    /// it and re-keying the guard on `default_greens.is_empty()` passes every
    /// remaining assertion here while making the empty-greens fixtures in
    /// `majit-metainterp` unrepresentable.
    #[test]
    fn merge_point_without_a_greens_key_is_refused() {
        fn expand(greens: proc_macro2::TokenStream) -> String {
            let config: JitInterpConfig = syn::parse2(quote! {
                state = S,
                env = Bytecode,
                #greens
                state_fields = { acc: int },
            })
            .expect("fixture attribute must parse");
            let func: ItemFn = parse_quote! {
                fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                    let mut driver: majit_metainterp::JitDriver<S> =
                        majit_metainterp::JitDriver::new(threshold);
                    let mut pc: usize = 0;
                    let mut state = S { acc: 0 };
                    while pc < program.len() {
                        jit_merge_point!(driver, program, pc; state);
                        let op = program[pc];
                        pc += 1;
                        match op {
                            0 => { state.acc += 1; }
                            _ => break,
                        }
                    }
                    state.acc
                }
            };
            transform_jit_interp(config, func).to_string()
        }

        let without_greens = expand(quote! {});
        assert!(
            without_greens.contains("compile_error"),
            "an attribute with no `greens` key must expand to a compile_error; \
             without one the merge point reports no concrete pc and the tracing \
             walk never advances. Expansion was:\n{without_greens}"
        );
        assert!(
            without_greens.contains("greens"),
            "the refusal must name `greens` so the reader knows what to add:\n\
             {without_greens}"
        );

        let with_greens = expand(quote! { greens = [pc, program], });
        assert!(
            !with_greens.contains("compile_error"),
            "the same interpreter with `greens = [pc, program]` must expand \
             cleanly. A compile_error here means the fixture itself is broken, \
             which would make the assertions above vacuous. Expansion was:\n\
             {with_greens}"
        );

        let empty_greens = expand(quote! { greens = [], });
        assert!(
            !empty_greens.contains("compile_error"),
            "`greens = []` is a DECLARATION of emptiness and must expand \
             cleanly. The refusal keys on the absence of the key, not on the \
             emptiness of the list: `majit-metainterp`'s dispatch-IR fixtures \
             spell `greens = []` precisely to grade the empty-greens encoding \
             (`num_green_args == 0`, and no `BC_*_GUARD_VALUE` in the prefix, \
             mirroring `jtransform.py` `promote_greens`), so refusing \
             this spelling deletes the only tests of the state the refusal \
             above is about. Expansion was:\n{empty_greens}"
        );
    }

    /// The refusal above reaches EVERY merge-point form, the bare
    /// `jit_merge_point!()` included.
    ///
    /// THIS ASSERTION USED TO RUN THE OTHER WAY, and the inversion is the
    /// record of a real change rather than a tidy-up. The guard was
    /// `args.state.is_some() && default_greens.is_empty()`, and the first
    /// conjunct confined it to a form a greens-less driver cannot be written
    /// in — so it never fired on a crate that needed it. This test pinned that
    /// gap, said in its own text that dropping the conjunct must invert the
    /// second assertion, and the conjunct is now gone. Anyone re-narrowing the
    /// guard reds here and has to invert it back deliberately.
    ///
    /// A `contains` assertion arms itself and a `!contains` one does not,
    /// so the two arms are no longer symmetric in what they prove. Both arms
    /// now expect the refusal, which means a fixture that stopped reaching
    /// this handler entirely would fail LOUDLY rather than pass quietly — the
    /// failure direction is the safe one. The `is_tracing()` count is kept on
    /// the bare arm anyway: it reads whether the bare form was processed at
    /// all, independently of what the guard decided, and it is the only thing
    /// here that can tell "refused" from "never expanded" if the refusal ever
    /// moves again.
    ///
    /// Do not fold this into the test above as a duplicate — that one varies
    /// the greens with the form fixed, this one varies the form with the
    /// greens fixed, and neither substitutes for the other.
    #[test]
    fn no_greens_refusal_reaches_every_merge_point_form() {
        fn expand(merge_point: proc_macro2::TokenStream) -> String {
            let config: JitInterpConfig = syn::parse2(quote! {
                state = S,
                env = Bytecode,
                state_fields = { acc: int },
            })
            .expect("fixture attribute must parse");
            let func: ItemFn = parse_quote! {
                fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                    let mut driver: majit_metainterp::JitDriver<S> =
                        majit_metainterp::JitDriver::new(threshold);
                    let mut pc: usize = 0;
                    let mut state = S { acc: 0 };
                    while pc < program.len() {
                        #merge_point
                        let op = program[pc];
                        pc += 1;
                        match op {
                            0 => { state.acc += 1; }
                            _ => break,
                        }
                    }
                    state.acc
                }
            };
            transform_jit_interp(config, func).to_string()
        }

        let state_form = expand(quote! { jit_merge_point!(driver, program, pc; state); });
        assert!(
            state_form.contains("compile_error"),
            "the `; state` arm is this test's positive control: with no greens \
             it must still be refused, otherwise the bare-form assertion below \
             is measuring a detector that is not running. Expansion was:\n\
             {state_form}"
        );

        let bare_form = expand(quote! { jit_merge_point!(); });
        assert_eq!(
            bare_form.matches("is_tracing").count(),
            1,
            "arming for the assertion below, which is a `!contains` and cannot \
             arm itself. `is_tracing()` is emitted from exactly two places in \
             this crate, both of them the merge-point handler's own arms, so \
             one occurrence is the fixture's one merge point having been \
             processed. A zero here means the bare form never reached the \
             handler, and the absent `compile_error` below would then be \
             measuring nothing. Expansion was:\n{bare_form}"
        );
        assert!(
            bare_form.contains("compile_error"),
            "the bare form with no `greens` key must be refused too. This \
             assertion was inverted when the `args.state.is_some()` conjunct \
             was dropped; if it now fails, the guard has been re-narrowed to \
             some subset of the merge-point forms. Widen it back rather than \
             inverting this line — a form-dependent refusal cannot fire on the \
             crates that need it, which is what the conjunct did for as long as \
             it stood. Expansion was:\n{bare_form}"
        );
    }

    /// `ll_portal_runner` sits in front of the interpreter. A `; state`
    /// merge point is the only form that has reds to extract, so that is
    /// the form that must emit `function_entry_structured`.
    #[test]
    fn state_merge_point_emits_function_entry_door() {
        let config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut driver: majit_metainterp::JitDriver<S> =
                    majit_metainterp::JitDriver::new(threshold);
                let mut pc: usize = 0;
                let mut state = S { acc: 0 };
                while pc < program.len() {
                    jit_merge_point!(driver, program, pc; state);
                    let op = program[pc];
                    pc += 1;
                    match op {
                        0 => { state.acc += 1; }
                        _ => break,
                    }
                }
                state.acc
            }
        };
        let expanded = transform_jit_interp(config, func).to_string();
        assert!(
            expanded.contains("function_entry_structured"),
            "the generated portal must call the function-entry door before \
             the interpreter loop. Expansion was:\n{expanded}"
        );
        let door_at = expanded
            .find("function_entry_structured")
            .expect("door present");
        let pc_at = expanded.find("let mut pc").expect("pc binding");
        let driver_at = expanded.find("let mut driver").expect("driver binding");
        let state_at = expanded.find("let mut state").expect("state binding");
        assert!(
            pc_at < door_at && driver_at < door_at && state_at < door_at,
            "the door must sit after driver/pc/state are bound. Expansion was:\n{expanded}"
        );
        assert!(
            !expanded.contains("__mainloop_portal"),
            "locals before the loop keep the inline door. Expansion was:\n{expanded}"
        );
    }

    /// Loop first, and driver/pc/state/env are parameters: `ll_portal_runner`
    /// is its own function and the body is `__mainloop_portal`.
    #[test]
    fn function_entry_door_splits_into_a_runner_when_the_loop_is_the_first_statement() {
        let config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let func: ItemFn = parse_quote! {
            fn mainloop(
                mut driver: &mut majit_metainterp::JitDriver<S>,
                program: &Bytecode,
                state: &mut S,
                mut pc: usize,
            ) -> i64 {
                loop {
                    jit_merge_point!(driver, program, pc; *state);
                    let op = program[pc];
                    pc += 1;
                    match op {
                        0 => { state.acc += 1; }
                        _ => break,
                    }
                }
                state.acc
            }
        };
        let expanded = transform_jit_interp(config, func).to_string();
        assert!(
            expanded.contains("function_entry_runner"),
            "the runner calls the direct door. Expansion was:\n{expanded}"
        );
        assert!(
            expanded.contains("__mainloop_portal"),
            "the body keeps a portal function. Expansion was:\n{expanded}"
        );
        assert!(
            !expanded.contains("function_entry_structured"),
            "the split removes the inline door. Expansion was:\n{expanded}"
        );
        let runner_at = expanded.find("fn mainloop").expect("runner");
        let portal_at = expanded.find("fn __mainloop_portal").expect("portal");
        let door_at = expanded.find("function_entry_runner").expect("door");
        assert!(
            runner_at < door_at && door_at < portal_at,
            "the door runs in the runner, before the portal body. Expansion was:\n{expanded}"
        );
    }

    #[test]
    fn function_entry_door_sits_before_the_merge_point_loop() {
        let config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut driver: majit_metainterp::JitDriver<S> =
                    majit_metainterp::JitDriver::new(threshold);
                let mut pc: usize = 0;
                let mut state = S { acc: 0 };
                while pc < 1 {
                    let _setup_loop = ();
                    pc += 1;
                }
                while pc < program.len() {
                    let _dispatch_loop = ();
                    jit_merge_point!(driver, program, pc; state);
                    let op = program[pc];
                    pc += 1;
                    match op {
                        0 => { state.acc += 1; }
                        _ => break,
                    }
                }
                state.acc
            }
        };
        let expanded = transform_jit_interp(config, func).to_string();
        let door_at = expanded
            .find("function_entry_structured")
            .expect("door present");
        let setup_at = expanded.find("_setup_loop").expect("setup loop");
        assert!(
            setup_at < door_at,
            "the door must sit after a setup loop, not before it. Expansion was:\n{expanded}"
        );
    }

    /// A `*mut T` portal's function-entry door drains the bridge walk's
    /// single-pass finish, not only the back-edge latch. The take sits after
    /// the entry call and before the loop's own back edge.
    #[test]
    fn pointer_portal_door_drains_single_pass_finish() {
        fn assert_drain_before_back_edge(expanded: &str, entry_call: &str) {
            let entry_at = expanded
                .find(entry_call)
                .unwrap_or_else(|| panic!("missing {entry_call}. Expansion was:\n{expanded}"));
            let back_edge_at = expanded[entry_at..]
                .find("back_edge_structured")
                .map(|at| entry_at + at)
                .unwrap_or(expanded.len());
            let door = &expanded[entry_at..back_edge_at];
            let flag_at = door.find("take_single_pass_finish").unwrap_or_else(|| {
                panic!(
                    "{entry_call} door must take the single-pass flag before \
                     the back edge. Door was:\n{door}"
                )
            });
            let after_flag = flag_at + "take_single_pass_finish".len();
            assert!(
                !door[after_flag..].starts_with("_"),
                "the flag take must be its own call. Door was:\n{door}"
            );
            assert!(
                door[after_flag..].contains("take_single_pass_finish_ref"),
                "{entry_call} door must drain the ref finish. Door was:\n{door}"
            );
        }

        let inline_config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let inline: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> *mut u8 {
                let mut driver: majit_metainterp::JitDriver<S> =
                    majit_metainterp::JitDriver::new(threshold);
                let mut pc: usize = 0;
                let mut state = S { acc: 0 };
                while pc < program.len() {
                    jit_merge_point!(driver, program, pc; state);
                    let op = program[pc];
                    pc += 1;
                    match op {
                        0 => { state.acc += 1; }
                        _ => break,
                    }
                }
                std::ptr::null_mut()
            }
        };
        let inline_expanded = transform_jit_interp(inline_config, inline).to_string();
        assert_drain_before_back_edge(&inline_expanded, "function_entry_structured");

        let runner_config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let runner: ItemFn = parse_quote! {
            fn mainloop(
                mut driver: &mut majit_metainterp::JitDriver<S>,
                program: &Bytecode,
                state: &mut S,
                mut pc: usize,
            ) -> *mut u8 {
                loop {
                    jit_merge_point!(driver, program, pc; *state);
                    let op = program[pc];
                    pc += 1;
                    match op {
                        0 => { state.acc += 1; }
                        _ => break,
                    }
                }
                std::ptr::null_mut()
            }
        };
        let runner_expanded = transform_jit_interp(runner_config, runner).to_string();
        let portal_at = runner_expanded
            .find("fn __mainloop_portal")
            .expect("portal function");
        assert_drain_before_back_edge(&runner_expanded[..portal_at], "function_entry_runner");
        let runner_door = &runner_expanded[..portal_at];
        let run_at = runner_door
            .find("FunctionEntryRunner :: Run")
            .unwrap_or_else(|| panic!("Run arm missing. Door was:\n{runner_door}"));
        let run_arm = &runner_door[run_at..];
        assert!(
            run_arm.contains("take_single_pass_finish_ref"),
            "the Run arm must drain the single-pass finish before the portal. \
             Arm was:\n{run_arm}"
        );
    }

    /// `can_enter_jit` / the generated helper hash `original_boxes[:num_green_args]`
    /// (`pyjitpl.py` `reached_loop_header` / `warmstate.py` `JitCell.get_uhash`).
    /// `greens = [pc, program]` is two slots; a third `__green_slot2` is the
    /// retired layout that prepended the back-edge target in front of that list.
    #[test]
    fn green_key_uses_declared_greens_without_prepending_the_target() {
        let config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut driver: majit_metainterp::JitDriver<S> =
                    majit_metainterp::JitDriver::new(threshold);
                let mut pc: usize = 0;
                let mut state = S { acc: 0 };
                while pc < program.len() {
                    jit_merge_point!(driver, program, pc; state);
                    let op = program[pc];
                    pc += 1;
                    match op {
                        0 => {
                            can_enter_jit!(driver, 0usize, &mut state, program, || {});
                            pc = 0;
                            continue;
                        }
                        _ => break,
                    }
                }
                state.acc
            }
        };
        let expanded = transform_jit_interp(config, func).to_string();
        assert!(
            expanded.contains("__green_slot0"),
            "declared greens must occupy slot 0. Expansion was:\n{expanded}"
        );
        assert!(
            expanded.contains("__green_slot1"),
            "greens = [pc, program] occupies two slots. Expansion was:\n{expanded}"
        );
        assert!(
            !expanded.contains("__green_slot2"),
            "a third slot is the retired prepended-target layout. Expansion was:\n{expanded}"
        );
        assert!(
            expanded.contains("__majit_green_key_mainloop") && expanded.contains("; 2usize"),
            "`__majit_green_key_mainloop` must return two declared-green slots. \
             Expansion was:\n{expanded}"
        );
    }

    /// A function-scope `let mut` green is restored from
    /// `PortalResume` / `ContinueRunningNormallyArgs` and passed into
    /// `__merge_*`. A loop-head `let` of the same name is not a carried
    /// input. An immutable loop-head green is not assigned.
    #[test]
    fn crn_resume_assigns_mut_greens_from_the_banks() {
        let config: JitInterpConfig = syn::parse2(quote! {
            state = S,
            env = Bytecode,
            greens = [pc, stackok, kind, bm, program],
            state_fields = { acc: int },
        })
        .expect("fixture attribute must parse");
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut driver: majit_metainterp::JitDriver<S> =
                    majit_metainterp::JitDriver::new(threshold);
                let mut pc: usize = 0;
                let mut state = S { acc: 0 };
                let mut stackok = false;
                let mut kind = 0usize;
                let mut bm = 0i64;
                while pc < program.len() {
                    let mut stackok = false;
                    let bm = 0i64;
                    jit_merge_point!(driver, program, pc; state);
                    can_enter_jit!(
                        driver, pc, &mut state, program, || {}, pc, 0;
                        pc, stackok, kind, bm, program
                    );
                }
                state.acc
            }
        };
        let expanded = transform_jit_interp(config, func).to_string();
        assert!(
            expanded.contains("PortalResume"),
            "the merge point must rebuild the resume. Expansion was:\n{expanded}"
        );
        assert!(
            expanded.contains("GreenFromI64"),
            "mut greens must be assigned from the banks. Expansion was:\n{expanded}"
        );
        let merge_at = expanded
            .find("fn __merge_mainloop")
            .expect("merge function");
        let merge_sig_end = expanded[merge_at..].find(") ->").expect("merge signature");
        let merge_sig = &expanded[merge_at..merge_at + merge_sig_end];
        assert!(
            merge_sig.contains("kind : impl"),
            "unshadowed `kind` is a carried input of `__merge_*`. Signature was:\n{merge_sig}"
        );
        assert!(
            !merge_sig.contains("stackok") && !merge_sig.contains("bm"),
            "loop-head lets are not carried inputs. Signature was:\n{merge_sig}"
        );
        let kind_assigns = expanded.matches("kind =").count();
        assert!(
            kind_assigns >= 2,
            "kind is assigned from the resume at the merge point and the back edge. \
             Expansion was:\n{expanded}"
        );
    }
}
