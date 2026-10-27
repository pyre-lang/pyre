//! `ValueType` ↔ `SomeValue` shell projection helpers.
//!
//! RPython's `RPythonAnnotator.complete()` attaches a `SomeValue`
//! directly to each `Variable.annotation` slot on the flowgraph
//! (`rpython/annotator/annrpython.py`,
//! `rpython/flowspace/model.py: Variable.annotation`).  Pyre writes the
//! same lattice node into `Variable.annotation` via
//! `legacy_annotator::setbinding_with`, which
//! routes through [`valuetype_to_someshell`] to build the shell.
//!
//! `ValueType::Unknown` returns `None` from [`valuetype_to_someshell`]
//! — annotation gaps surface fail-loud at the rtyper's `bindingrepr`
//! instead of being bridged to a fabricated GC reference.

use std::collections::BTreeMap;

use crate::annotator::model::{KnownType, SomeFloat, SomeInstance, SomeInteger, SomeValue};
use crate::model::ValueType;

/// Classdef-less `Ref` shells are not counted in a thread-local.
/// A `SomeInstance` carries its `classdef` only when the caller has a
/// `Bookkeeper` (`Bookkeeper::getuniqueclassdef`). This projector does not.

fn record_ref_payload(payload: Option<&str>) {
    let _ = payload;
}

/// Classdef-less `SomeInstance` minted outside the `Ref` arm. The census
/// counter is gone; callers still name the site so a later count can return.
pub(crate) fn record_classdef_less_mint(site: &str, key: &str) {
    let _ = (site, key);
}

/// A startblock `Ref` input that kept the classdef-less shell.
pub(crate) fn record_classdef_less_input(
    graph: &str,
    var: &str,
    payload: Option<&str>,
    class_root: Option<&str>,
    canon: Option<&str>,
    raw_known: bool,
    canon_known: bool,
    has_bk: bool,
) {
    let _ = (
        graph,
        var,
        payload,
        class_root,
        canon,
        raw_known,
        canon_known,
        has_bk,
    );
}

/// Former classdef-less `Ref` census. The thread-local counter is gone.
pub(crate) fn dump_classdef_less_ref_census() {}

/// RPython `SomeValue` lattice projection of the legacy `ValueType`.
///
/// `RPythonTyper.bindingrepr` (in `rtyper.rs`) dispatches purely on the
/// `SomeValue` shape via `rtyper_makekey` / `rtyper_makerepr`.  `Int`
/// and `Float` resolve cleanly through `rint::IntegerRepr` /
/// `rfloat::FloatRepr`.
///
/// Mapping:
///
/// | legacy `ValueType` | `SomeValue` shell      | RPython source / status |
/// |--------------------|------------------------|--------------------------|
/// | `Int`              | `Integer(SomeInteger)` | `model.py` -> `SomeValue::Integer` arm. |
/// | `Float`            | `Float(SomeFloat)`     | `model.py` -> `SomeValue::Float` arm. |
/// | `Ref(_)`           | `Instance(SomeInstance{classdef:None,..})` | `model.py`.  Typed pointers should lift to `SomePtr(ll_ptrtype)` (`llannotation.py:64-70`), but the correct Ptr must come from the producer writing `Variable.annotation` directly — not from a process-global root-string lookup.  This fallback projection keeps all Ref variants classdef-less → `GcRef` via `rclass.py`. |  allow-line-citation
/// | `Void`             | `Impossible`           | `model.py:627` -> `SomeValue::Impossible` arm. |
/// | `State`            | `Instance(SomeInstance{classdef:None,..})` | **TODO: no upstream equivalent**.  Pyre-only `State` carries the JIT state pointer (a struct pointer to interpreter state).  RPython has no analogue; the `SomeInstance(classdef=None)` projection is a temporary fallback that lets the rtyper proceed without a real bookkeeper-attached pyre `ClassDef`.  Projects to `GcRef` via the same chain as `Ref`. |
/// | `Unknown`          | `None`                 | **Fail-loud — annotation gap with no annotation-stage shell.**  RPython's annotator never produces an unknown lattice node — every Variable is annotated with a definite `SomeValue`, and unreachable code stays at `SomeImpossible`.  Pyre's `Unknown` is a coverage gap (annotator did not narrow / producer did not call `set_some`).  Returning `None` leaves `Variable.annotation` empty so `bindingrepr` panics with `KeyError: no binding for arg` (`annotator/annrpython.rs`'s `binding`) on the first attempt to lower the affected `Variable`, surfacing the producer-side gap rather than silently bridging it to `GcRef` via a fabricated `SomeInstance(None)` shell — that bridging conflated an *annotation-stage* lattice node with the **legacy** `resolve_types(Unknown) -> ConcreteType::Unknown -> GcRef` resolver-stage backfill. |
///
/// Returns `None` only for `ValueType::Unknown`; every other variant
/// projects to a definite `SomeValue` shell.
pub fn valuetype_to_someshell(vt: &ValueType) -> Option<SomeValue> {
    match vt {
        // `Int` shells to `SomeInteger { unsigned: false }` (default);
        // `Unsigned` shells to `SomeInteger { unsigned: true }` so the
        // rtyper picks `IntegerRepr.lowleveltype = Unsigned`
        // (`rint.py:_init_repr`).  `getkind(Unsigned) == 'int'` so the
        // codewriter / regalloc share register classes via Int|Unsigned
        // arms downstream.
        ValueType::Int => Some(SomeValue::Integer(SomeInteger::default())),
        // `SomeSingleFloat`, agreeing with the bookkeeper, which already
        // shells a Rust `f32` field this way (`bookkeeper.rs`'s `"f32"`
        // arm).  Shelling it as `SomeInteger` made the two disagree
        // about the same type.
        ValueType::SingleFloat => Some(SomeValue::SingleFloat(
            crate::annotator::model::SomeSingleFloat::new(),
        )),
        ValueType::Unsigned => Some(SomeValue::Integer(SomeInteger::new(false, true))),
        ValueType::Int128 => Some(SomeValue::Integer(SomeInteger::new_with_knowntype(
            false,
            KnownType::LongLongLong,
        ))),
        ValueType::UInt128 => Some(SomeValue::Integer(SomeInteger::new_with_knowntype(
            true,
            KnownType::ULongLongLong,
        ))),
        // RPython `SomeBool` (`annotator/model.py`) is a
        // distinct lattice node from `SomeInteger`; the rtyper picks
        // `BoolRepr` (`rmodel.rs::BoolRepr`) which lowers to LL `Bool`
        // (integer-compatible).  Until a richer Bool annotation lands
        // (with truthy_value tracking per `model.py:188-194`), shape it
        // as `SomeBool::default` matching `SomeBool()` upstream.
        ValueType::Bool => Some(SomeValue::Bool(crate::annotator::model::SomeBool::default())),
        ValueType::Float => Some(SomeValue::Float(SomeFloat::default())),
        // A `str`/`String`/`Wtf8` value shells to `SomeString`
        // (`annotator/model.py` `SomeString`), the widest string shell
        // (not-const, may-contain-nul), so a string-typed struct field
        // seeds a string attr that unions cleanly with the value written
        // to it instead of the classdef-less `SomeInstance(None)` the
        // `Ref` fallback yields.
        ValueType::Str => Some(SomeValue::String(crate::annotator::model::SomeString::new(
            false, false,
        ))),
        // RPython `StringBuilder()` binds to `SomeStringBuilder` at the
        // annotator (`rlib/rstring.py`); the rtyper then resolves it
        // to `StringBuilderRepr`.  Carries no payload — the method call
        // surface (`append` / `build` / `getlength`) lives on
        // `SomeStringBuilder` itself.
        ValueType::StringBuilder => Some(SomeValue::StringBuilder(
            crate::annotator::model::SomeStringBuilder::new(),
        )),
        ValueType::Ref(payload) => {
            // RPython typed pointers lift to `SomePtr(ll_ptrtype)`
            // (`llannotation.py:64-70`), but the correct Ptr must come
            // from the producer (annotator with bookkeeper / host-class
            // registry) writing `Variable.annotation` directly — not
            // from a process-global root-string lookup here.  A global
            // bare-name index conflates cross-module same-name structs
            // and bypasses RPython's object-identity-based lltype cache.
            // The MIR front-end does not attach a per-`&Foo`-input
            // lltype `Ptr` to `Variable.annotation` (the faithful
            // counterpart of `lltype.py
            // _ptrEntry.compute_annotation`), so every `Ref` input
            // shells to the classdef-less `SomeInstance` here rather
            // than fabricating a `Ptr` from a bare name.  Routing a
            // host-registry lookup back into *this* shell projector
            // would be the rejected bare-name path.
            record_ref_payload(payload.as_deref());
            Some(ref_fallback_instance())
        }
        ValueType::State => {
            // TODO: no upstream equivalent.  Pyre's `State`
            // carries the JIT state pointer; RPython has no analogue.
            // `SomeInstance(classdef=None)` is a temporary fallback
            // that lets the rtyper resolve to `GcRef` without a real
            // bookkeeper-attached pyre `ClassDef`.
            record_classdef_less_mint("valuetype_to_someshell", "State");
            Some(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                BTreeMap::new(),
            )))
        }
        ValueType::Unknown => {
            // Fail-loud — annotation gap, NOT annotation-stage parity.
            // RPython's annotator never produces an unknown lattice
            // node; pyre's `Unknown` is a coverage gap.  Returning
            // `None` leaves `Variable.annotation` empty so the rtyper
            // panics at `bindingrepr` (`annotator/annrpython.rs`'s `binding`,
            // "KeyError: no binding for arg") on the first attempt to
            // lower an Unknown Variable.  This surfaces the producer-
            // side gap rather than silently bridging it to `GcRef` via
            // a fabricated `SomeInstance(None)` shell — that bridging
            // conflated the annotation-stage lattice node with the
            // **legacy** resolver-stage backfill
            // (`resolve_types(Unknown) -> ConcreteType::Unknown ->
            // GcRef`).  Convergence path: precise producer-side
            // `set_some` for every `Variable`.
            None
        }
        ValueType::Void => Some(SomeValue::Impossible),
    }
}

fn ref_fallback_instance() -> SomeValue {
    SomeValue::Instance(SomeInstance::new(None, false, BTreeMap::new()))
}

/// Reduce a `SomeValue` lattice node to its `ValueType` discriminator.
/// Inverse of [`valuetype_to_someshell`].
///
/// RPython parity: `getkind` family in `rpython/rtyper/lltypesystem/lltype.py`
/// reduces lltypes to backend kinds; the analogue here reduces
/// annotation-stage `SomeValue` to pyre's flat `ValueType` enum used by
/// downstream `codewriter` consumers that haven't been ported to
/// `SomeValue` directly.
pub fn somevalue_to_valuetype(s: &SomeValue) -> ValueType {
    match s {
        SomeValue::Integer(integer) => match integer.base.knowntype {
            KnownType::LongLongLong => ValueType::Int128,
            KnownType::ULongLongLong => ValueType::UInt128,
            _ => ValueType::Int,
        },
        SomeValue::Bool(_) => ValueType::Bool,
        SomeValue::Float(_) | SomeValue::LongFloat(_) => ValueType::Float,
        // Preserve the annotation-level type across fixpoint rounds.  The
        // later codewriter kind projection banks SingleFloat as an integer;
        // collapsing it here would instead union Int with SingleFloat on the
        // next annotator pass and erase the binding as Unknown.
        SomeValue::SingleFloat(_) => ValueType::SingleFloat,
        // Classed `SomeInstance` keeps the classdef name so
        // `valuetype_to_concrete_for_legacy` can apply `rclass.py
        // getgcflavor`. Classless is the root instance
        // (`rclass.py getinstancerepr(None)`), flavor `'gc'`.
        SomeValue::Instance(inst) => {
            ValueType::Ref(inst.classdef.as_ref().map(|cd| cd.borrow().name.clone()))
        }
        SomeValue::Ptr(_) | SomeValue::PBC(_) => ValueType::Ref(None),
        // Keep the `StringBuilder` shell distinct across the roundtrip so
        // the rtyper picks `StringBuilderRepr` rather than the generic
        // `Ref` fallback (which would erase the builder method surface).
        SomeValue::StringBuilder(_) => ValueType::StringBuilder,
        // `SomeImpossibleValue` represents unreachable code (`model.py:627`),
        // which projects to `ValueType::Void` in pyre's flat enum just
        // like upstream `lltype.Void`.
        SomeValue::Impossible => ValueType::Void,
        // Other variants (String / List / Tuple / Dict / Iterator /
        // Exception / None_ / Property / InteriorPtr / LLADTMeth /
        // Builtin / BuiltinMethod / WeakRef / TypeOf / ByteArray /
        // Char / UnicodeCodePoint / UnicodeString / Type / Object) have
        // no direct pyre `ValueType` mapping. Consumers that need the
        // precise lattice node should read the `Variable.annotation`
        // shell directly; this reduced projection falls back to `Ref`
        // so GC-pointer lowering applies.
        _ => ValueType::Ref(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_ref_uses_instance_fallback() {
        let shell = valuetype_to_someshell(&ValueType::Ref(Some("SomeStruct".to_string())))
            .expect("Ref projects");
        match shell {
            SomeValue::Instance(inst) => {
                assert!(inst.classdef.is_none());
                assert!(!inst.can_be_none);
                assert!(inst.flags.is_empty());
            }
            other => panic!("typed Ref must use fallback Instance, got {other:?}"),
        }
    }

    #[test]
    fn typed_ref_payload_is_not_looked_up_in_a_global_registry() {
        // The projector must keep discarding `Ref(Some(root))`.  A
        // process-global bare-name index is the rejected path; the
        // producer attaches the precise `Variable.annotation` instead.
        let named = valuetype_to_someshell(&ValueType::Ref(Some("RootScope".to_string())))
            .expect("named Ref projects");
        let opaque = valuetype_to_someshell(&ValueType::Ref(None)).expect("opaque Ref projects");
        match (named, opaque) {
            (SomeValue::Instance(named), SomeValue::Instance(opaque)) => {
                assert!(named.classdef.is_none());
                assert!(opaque.classdef.is_none());
            }
            other => panic!("both Ref payloads must stay classdef-less, got {other:?}"),
        }
    }

    #[test]
    fn stringbuilder_roundtrips_through_shell() {
        // `ValueType::StringBuilder` must project to the `SomeStringBuilder`
        // shell (so the rtyper picks `StringBuilderRepr`) and reduce back to
        // `ValueType::StringBuilder` (so the roundtrip is stable and the
        // builder is never erased to the generic `Ref` fallback).
        let shell =
            valuetype_to_someshell(&ValueType::StringBuilder).expect("StringBuilder projects");
        assert!(
            matches!(shell, SomeValue::StringBuilder(_)),
            "StringBuilder must project to SomeStringBuilder, got {shell:?}"
        );
        assert_eq!(somevalue_to_valuetype(&shell), ValueType::StringBuilder);
    }

    #[test]
    fn singlefloat_roundtrips_through_shell() {
        let shell = valuetype_to_someshell(&ValueType::SingleFloat)
            .expect("SingleFloat projects to an annotation shell");
        assert!(matches!(shell, SomeValue::SingleFloat(_)));
        assert_eq!(somevalue_to_valuetype(&shell), ValueType::SingleFloat);
    }

    #[test]
    fn uint128_shell_is_nonnegative() {
        let shell = valuetype_to_someshell(&ValueType::UInt128).expect("UInt128 projects");
        let SomeValue::Integer(integer) = shell else {
            panic!("UInt128 must project to SomeInteger");
        };
        assert!(integer.nonneg);
        assert_eq!(integer.base.knowntype, KnownType::ULongLongLong);
    }
}
