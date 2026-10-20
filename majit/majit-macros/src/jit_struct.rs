//! `#[jit_struct]` attribute macro — generic GcCache descriptor registration.
//!
//! RPython parity: descr.py `get_size_descr` + descr.py
//! `get_field_descr`. RPython's translator walks `lltype.Struct` definitions
//! and emits FieldDescr/SizeDescr automatically from the type layout; this
//! macro plays the same auto-discovery role for Rust structs.
//!
//! ```ignore
//! #[jit_struct]
//! struct Node {
//!     value: i64,
//!     next: Option<Box<Node>>,
//! }
//!
//! let descr = Node::__majit_register_descrs(&mut gc_cache);
//! ```

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type};

pub(crate) fn expand(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input: DeriveInput = match syn::parse2(item) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error(),
    };

    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => named,
            _ => {
                return syn::Error::new_spanned(
                    &input.ident,
                    "#[jit_struct] only supports structs with named fields",
                )
                .to_compile_error();
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &input.ident,
                "#[jit_struct] can only be applied to structs",
            )
            .to_compile_error();
        }
    };

    let struct_name = &input.ident;
    let generics = &input.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let field_names: Vec<String> = fields
        .named
        .iter()
        .map(|f| f.ident.as_ref().unwrap().to_string())
        .collect();

    let field_registrations = fields.named.iter().enumerate().map(|(idx, field)| {
        let fname = field.ident.as_ref().unwrap();
        let fname_str = fname.to_string();
        let fty = &field.ty;
        let (ir_type_tok, flag_tok) = classify_field_type(fty);
        quote! {
            let _ = gc_cache.get_field_descr(
                __majit_key.clone(),
                #fname_str,
                None,
                ::std::mem::offset_of!(Self, #fname),
                ::std::mem::size_of::<#fty>(),
                #ir_type_tok,
                false,
                false,
                #flag_tok,
                u32::MAX,
                false,
                // The declaration order of the struct's own named fields — a
                // position this macro computes, never an unresolved claim.
                Some(#idx),
                false,
            );
        }
    });

    quote! {
        #input

        impl #impl_generics #struct_name #ty_generics #where_clause {
            /// descr.py:109 cache key identity for this struct.
            ///
            /// Deterministic within a process; not stable across builds.
            /// Analogue of RPython's `lltype.Struct` Python-object identity.
            ///
            /// Implementation: `path_hash_stripped_crate` drops the
            /// leading `<crate>::` segment from `module_path!()` before
            /// hashing so analyzer-time / build.rs (`module_path_from_
            /// source_file`) / hard-coded `build_object_descr_group_with_
            /// def_path` def-paths and the macro-emitted runtime
            /// registration share a single `path_hash` namespace.  PyPy
            /// has no crate boundary in its `lltype.Struct` identity
            /// (`descr.py cache[STRUCT]`); the strip aligns pyre
            /// with that convention.
            pub fn __majit_type_id() -> u64 {
                ::majit_ir::descr::path_hash_stripped_crate(
                    module_path!(),
                    stringify!(#struct_name),
                )
            }

            /// Field names declared at macro-expansion time, in declaration order.
            pub const __MAJIT_FIELD_NAMES: &'static [&'static str] = &[
                #( #field_names ),*
            ];

            /// descr.py get_size_descr + heaptracker.all_fielddescrs auto-discovery.
            ///
            /// Registers this struct's SizeDescr plus one FieldDescr per named
            /// field into `gc_cache`. Idempotent via the `_cache_size` /
            /// `_cache_field` cache hits. Returns the SizeDescr.
            ///
            /// The cache key is derived from the crate-stripped module path
            /// hashed to u64; that is the cache-lookup identity only. The
            /// numeric GC tid is stamped later by
            /// `register_unresolved_struct_tids` (`gc.py init_size_descr` /
            /// `gctypelayout.py get_type_id`), so the u64 hash collision
            /// risk does not translate into a tid collision.
            pub fn __majit_register_descrs(
                gc_cache: &mut ::majit_ir::descr::GcCache,
            ) -> ::majit_ir::descr::DescrRef {
                let __majit_type_id = Self::__majit_type_id();
                let __majit_key = ::majit_ir::descr::LLType::struct_key(__majit_type_id);
                let __majit_size_descr = gc_cache.get_size_descr(
                    __majit_key.clone(),
                    ::std::mem::size_of::<Self>(),
                    0,
                    false,
                );
                #( #field_registrations )*
                __majit_size_descr
            }
        }
    }
}

/// Classify a Rust field type into (IR Type, ArrayFlag).
///
/// RPython parity: descr.py `get_type_flag(FIELDTYPE)`:
///   - `lltype.Ptr` with `_gckind == 'gc'`         → FLAG_POINTER  (Ref)
///   - `lltype.Ptr` otherwise (raw pointer)         → FLAG_UNSIGNED (Int)
///   - `lltype.Struct`                              → FLAG_STRUCT
///   - `lltype.Float`                               → FLAG_FLOAT
///   - `lltype.SingleFloat` (`f32`)                 → FLAG_UNSIGNED
///     (not `lltype.Float`; the signed `rffi.cast` test fails)
///   - signed `lltype.Number` (cast(-1) == -1)      → FLAG_SIGNED
///   - otherwise                                    → FLAG_UNSIGNED
///
/// Rust mapping limitations:
///   - Nested structs (`foo: BarStruct`) cannot be distinguished from
///     opaque types at proc-macro time without full type resolution.
///     They fall through to Ref+Pointer, matching the GC-managed-pointer
///     default; use a manual `#[jit_struct]` on the inner type if the
///     field should participate in layout discovery.
///   - longlong is assumed equivalent to i64 on the 64-bit targets majit
///     supports; mapping it to `FLAG_FLOAT` mirrors RPython's
///     `is_longlong(TYPE)` branch for 32-bit back-ends that would not
///     compile here anyway.
fn classify_field_type(ty: &Type) -> (TokenStream, TokenStream) {
    // Raw pointers: `*const T` / `*mut T`.
    // descr.py — non-gc Ptr → FLAG_UNSIGNED (kept as Int bank because
    // RPython stores the raw address in a signed-int slot).
    if matches!(ty, Type::Ptr(_)) {
        return (
            quote! { ::majit_ir::value::Type::Int },
            quote! { ::majit_ir::descr::ArrayFlag::Unsigned },
        );
    }
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && tp.path.segments.len() == 1
    {
        let ident = &tp.path.segments[0].ident;
        let s = ident.to_string();
        match s.as_str() {
            "i8" | "i16" | "i32" | "i64" | "isize" => {
                return (
                    quote! { ::majit_ir::value::Type::Int },
                    quote! { ::majit_ir::descr::ArrayFlag::Signed },
                );
            }
            "u8" | "u16" | "u32" | "u64" | "usize" | "bool" => {
                return (
                    quote! { ::majit_ir::value::Type::Int },
                    quote! { ::majit_ir::descr::ArrayFlag::Unsigned },
                );
            }
            // `get_type_flag(SingleFloat)` is FLAG_UNSIGNED: not `lltype.Float`,
            // and `rffi.cast(TYPE, -1) == -1` fails. Size stays `size_of::<f32>()`.
            "f32" => {
                return (
                    quote! { ::majit_ir::value::Type::Int },
                    quote! { ::majit_ir::descr::ArrayFlag::Unsigned },
                );
            }
            "f64" => {
                return (
                    quote! { ::majit_ir::value::Type::Float },
                    quote! { ::majit_ir::descr::ArrayFlag::Float },
                );
            }
            _ => {}
        }
    }
    (
        quote! { ::majit_ir::value::Type::Ref },
        quote! { ::majit_ir::descr::ArrayFlag::Pointer },
    )
}
