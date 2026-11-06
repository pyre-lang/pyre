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

use crate::annotator::model::{
    DescKind, KnownType, SomeFloat, SomeInstance, SomeInteger, SomeValue, s_str,
};
use crate::model::ValueType;
use crate::translator::rtyper::lltypesystem::lltype::PtrTarget;

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
    if crate::translator::rtyper::cutover::rtyper_verbose_enabled() {
        eprintln!(
            "[classdef-less input] graph={graph} var={var} payload={payload:?} \
             class_root={class_root:?} canon={canon:?} raw_known={raw_known} \
             canon_known={canon_known} has_bk={has_bk}"
        );
    }
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
/// | `Ref(_)`           | `Instance(SomeInstance{classdef:None,..})` or `SomeString` for a str-family payload | `model.py`.  Typed pointers should lift to `SomePtr(ll_ptrtype)` (`llannotation.py` `lltype_to_annotation`), but the correct Ptr must come from the producer writing `Variable.annotation` directly — not from a process-global root-string lookup.  A `Ref` whose payload is `is_str_family_class_root` (`String`, `str`, `Wtf8Buf`, …) is `SomeString` with `no_nul=False` (`annotator/model.py` `SomeString.__init__` default); `no_nul` needs `rlib/rstring.py` `assert_str0`. `rstr.py` has one string type.  Other Ref variants stay classdef-less → `GcRef` via `rclass.py`. |
/// | `Void`             | `Impossible`           | `model.py:627` -> `SomeValue::Impossible` arm. |
/// | `State`            | `Instance(SomeInstance{classdef:None,..})` | **TODO: no upstream equivalent**.  Pyre-only `State` carries the JIT state pointer (a struct pointer to interpreter state).  RPython has no analogue; the `SomeInstance(classdef=None)` projection is a temporary fallback that lets the rtyper proceed without a real bookkeeper-attached pyre `ClassDef`.  Projects to `GcRef` via the same chain as `Ref`. |
/// | `Unknown`          | `None`                 | **Fail-loud — annotation gap with no annotation-stage shell.**  RPython's annotator never produces an unknown lattice node — every Variable is annotated with a definite `SomeValue`, and unreachable code stays at `SomeImpossible`.  Pyre's `Unknown` is a coverage gap (annotator did not narrow / producer did not call `set_some`).  Returning `None` leaves `Variable.annotation` empty so `bindingrepr` panics with `KeyError: no binding for arg` (`annotator/annrpython.rs`'s `binding`) on the first attempt to lower the affected `Variable`, surfacing the producer-side gap rather than silently bridging it to `GcRef` via a fabricated `SomeInstance(None)` shell. `resolve_types` likewise panics on leftover Unknown (`rtyper.py setconcretetype`). |
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
            // A rust-heap `String` / `&str` / `Wtf8Buf` is still
            // `ValueType::Ref` so `ll_strconcat` does not consume a
            // `{cap, ptr, len}` header (`tyref_to_value_type`). The
            // annotation is `SomeString` with `no_nul=False`
            // (`annotator/model.py` `SomeString.__init__` default):
            // `rstr.py` has one string type. `no_nul` needs
            // `rlib/rstring.py` `assert_str0`. `binaryop.py`
            // `pairtype(SomeObject, SomeObject).union` raises, so a
            // classdef-less `SomeInstance` here cannot join
            // `String::new()`.
            if let Some(shell) = str_family_ref_shell(payload.as_deref()) {
                return Some(shell);
            }
            // RPython typed pointers lift to `SomePtr(ll_ptrtype)`
            // (`llannotation.py` `lltype_to_annotation`), but the
            // correct Ptr must come from the producer (annotator with
            // bookkeeper / host-class registry) writing
            // `Variable.annotation` directly — not from a
            // process-global root-string lookup here.  A global
            // bare-name index conflates cross-module same-name structs
            // and bypasses RPython's object-identity-based lltype cache.
            // The MIR front-end does not attach a per-`&Foo`-input
            // lltype `Ptr` to `Variable.annotation` (the faithful
            // counterpart of `lltype.py`
            // `_ptrEntry.compute_annotation`), so every other `Ref`
            // input shells to the classdef-less `SomeInstance` here
            // rather than fabricating a `Ptr` from a bare name.
            // Routing a host-registry lookup back into *this* shell
            // projector would be the rejected bare-name path.
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
            // **legacy** resolver-stage Unknown leftover
            // (`rtyper.py setconcretetype` / `bindingrepr`).
            // Convergence path: precise producer-side `set_some`
            // for every `Variable`.
            None
        }
        ValueType::Void => Some(SomeValue::Impossible),
    }
}

fn ref_fallback_instance() -> SomeValue {
    SomeValue::Instance(SomeInstance::new(None, false, BTreeMap::new()))
}

/// `SomeString` with `no_nul=False` (`SomeString.__init__` default) when
/// `payload` is `is_str_family_class_root`. `no_nul` needs
/// `rlib/rstring.py` `assert_str0`.
fn str_family_ref_shell(payload: Option<&str>) -> Option<SomeValue> {
    payload
        .filter(|root| {
            crate::front::mir::is_str_family_class_root(root)
                || crate::front::mir::is_cow_str_family_class_root(root)
        })
        .map(|_| s_str())
}

/// Reduce a `SomeValue` lattice node to its `ValueType` discriminator.
/// Inverse of [`valuetype_to_someshell`].
///
/// RPython parity: `getkind` family in `rpython/rtyper/lltypesystem/lltype.py`
/// reduces lltypes to backend kinds; the analogue here reduces
/// annotation-stage `SomeValue` to pyre's flat `ValueType` enum used by
/// downstream `codewriter` consumers that haven't been ported to
/// `SomeValue` directly.
///
/// `Unsigned` shells to `SomeInteger { unsigned: true }`
/// (`SomeInteger::new`); this inverse returns `Unsigned` when
/// `integer.unsigned` (the 128-bit knowntypes stay `Int128` /
/// `UInt128`). `Str` shells to `SomeString` (`SomeString::new` /
/// `SomeString.__init__` default). A str-family `Ref(Some(..))`
/// (`is_str_family_class_root`) shells to the same `SomeString`
/// (`s_str`); both read back as `Str` because `rstr.py` has one
/// string type and `Str` is the dedicated discriminator. The legacy
/// walker re-reads the slot via `read_binding`, so a second
/// `Unsigned` / `Str` incoming is then the same-shell union
/// (`binaryop.py` `pairtype(SomeInteger, SomeInteger).union` of two
/// unsigned integers; `pairtype(SomeString, SomeString).union`).
pub fn somevalue_to_valuetype(s: &SomeValue) -> ValueType {
    match s {
        SomeValue::Integer(integer) => match integer.base.knowntype {
            KnownType::LongLongLong => ValueType::Int128,
            KnownType::ULongLongLong => ValueType::UInt128,
            _ if integer.unsigned => ValueType::Unsigned,
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
        // `history.getkind` / `codewriter/support.py` (via `getkind`): a
        // `Ptr` whose `TO` is `FuncType` (`FuncType._gckind == 'raw'`) is
        // `'int'`. GC pointees stay `Ref`. Raw non-function pointees stay
        // `Ref` here; that kind is a separate decision.
        SomeValue::Ptr(p) => match &p.ll_ptrtype.TO {
            PtrTarget::Func(_) => ValueType::Int,
            _ => ValueType::Ref(None),
        },
        // Frozen / class PBCs stay `Ref`. A function-PBC's
        // `FunctionsPBCRepr.lowleveltype` is `Ptr(FuncType)` even when
        // `can_be_None` (`rtype_is_None` → `ptr_iszero`); `history.getkind`
        // of that pointer is `'int'`. Projecting `Ref` here makes the
        // skip-path overlay `GcRef` while `FUNC.RESULT` is `'i'`.
        SomeValue::PBC(pbc)
            if !pbc.descriptions.is_empty()
                && pbc
                    .descriptions
                    .values()
                    .all(|d| d.kind() == DescKind::Function) =>
        {
            ValueType::Int
        }
        SomeValue::PBC(_) => ValueType::Ref(None),
        // Keep the `StringBuilder` shell distinct across the roundtrip so
        // the rtyper picks `StringBuilderRepr` rather than the generic
        // `Ref` fallback (which would erase the builder method surface).
        SomeValue::StringBuilder(_) => ValueType::StringBuilder,
        // Inverse of `valuetype_to_someshell(Str)` and of a str-family
        // `Ref` (`s_str`). `rstr.py` has one string type.
        SomeValue::String(_) => ValueType::Str,
        // `SomeImpossibleValue` represents unreachable code (`model.py`),
        // which projects to `ValueType::Void` in pyre's flat enum just
        // like upstream `lltype.Void`.
        SomeValue::Impossible => ValueType::Void,
        // Other variants (List / Tuple / Dict / Iterator /
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
    fn str_family_ref_annotates_somestring_and_unions_with_string_new() {
        // `is_str_family_class_root`: a rust-heap `String`, `&str`, or
        // `Wtf8Buf` is `SomeString` with `no_nul=False`
        // (`annotator/model.py` `SomeString.__init__` default). `no_nul`
        // needs `rlib/rstring.py` `assert_str0`. `String::new()` is
        // `SomeString` (`string_constructor`). `binaryop.py`
        // `pairtype(SomeObject, SomeObject).union` raises on String ∪
        // Instance, so the classdef-less `Ref` fallback cannot join
        // that constructor.
        use crate::annotator::model::{SomeString, union};
        let s_new = SomeValue::String(SomeString::new(false, false));
        for root in ["string::String", "String", "str", "Wtf8Buf"] {
            let shell = valuetype_to_someshell(&ValueType::Ref(Some(root.to_string())))
                .unwrap_or_else(|| panic!("{root} Ref must project"));
            let SomeValue::String(s) = &shell else {
                panic!("str-family Ref({root}) must be SomeString, got {shell:?}");
            };
            assert!(
                !s.inner.no_nul,
                "str-family Ref({root}) is SomeString(no_nul=False) (SomeString.__init__ default)"
            );
            let joined = union(&shell, &s_new).unwrap_or_else(|e| {
                panic!("str-family Ref({root}) must union with String::new(): {e}")
            });
            assert!(
                matches!(joined, SomeValue::String(_)),
                "str-family Ref({root}) ∪ String::new() must stay SomeString, got {joined:?}"
            );
        }
        let unnamed = valuetype_to_someshell(&ValueType::Ref(None)).expect("opaque Ref projects");
        assert!(
            matches!(&unnamed, SomeValue::Instance(inst) if inst.classdef.is_none()),
            "unnamed Ref stays classdef-less, got {unnamed:?}"
        );
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
    fn function_pointer_someptr_projects_as_int() {
        let s = crate::annotator::bookkeeper::raw_fn_ptr_somevalue();
        assert_eq!(somevalue_to_valuetype(&s), ValueType::Int);
    }

    #[test]
    fn function_pbc_projects_as_int() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::description::FunctionDesc;
        use crate::annotator::model::SomePBC;
        use crate::flowspace::argument::Signature;
        use std::cell::RefCell;
        use std::rc::Rc;

        let bk = Rc::new(Bookkeeper::new());
        let fd = Rc::new(RefCell::new(FunctionDesc::new(
            bk,
            None,
            "app_profile_call",
            Signature::new(vec![], None, None),
            None,
            None,
        )));
        let s = SomeValue::PBC(SomePBC::new(
            vec![crate::annotator::description::DescEntry::function(fd)],
            true,
        ));
        assert_eq!(somevalue_to_valuetype(&s), ValueType::Int);
    }

    #[test]
    fn unsigned_roundtrips_through_shell() {
        // `valuetype_to_someshell(Unsigned)` is `SomeInteger(unsigned=True)`
        // (`SomeInteger::new`); `binaryop.py`
        // `pairtype(SomeInteger, SomeInteger).union` of two unsigned
        // integers succeeds, so the inverse must report `Unsigned`.
        let shell = valuetype_to_someshell(&ValueType::Unsigned).expect("Unsigned projects");
        let SomeValue::Integer(integer) = &shell else {
            panic!("Unsigned must project to SomeInteger, got {shell:?}");
        };
        assert!(integer.unsigned);
        assert_eq!(somevalue_to_valuetype(&shell), ValueType::Unsigned);
    }

    #[test]
    fn str_roundtrips_through_shell() {
        // `valuetype_to_someshell(Str)` is `SomeString` (`SomeString.__init__`
        // default). A str-family `Ref(Some(..))` (`is_str_family_class_root`)
        // shells to the same `SomeString` (`s_str`); both read back as
        // `Str` because `rstr.py` has one string type.
        let shell = valuetype_to_someshell(&ValueType::Str).expect("Str projects");
        assert!(
            matches!(shell, SomeValue::String(_)),
            "Str must project to SomeString, got {shell:?}"
        );
        assert_eq!(somevalue_to_valuetype(&shell), ValueType::Str);
        let family =
            valuetype_to_someshell(&ValueType::Ref(Some("Wtf8Buf".to_string()))).expect("family");
        assert!(
            matches!(family, SomeValue::String(_)),
            "str-family Ref must project to SomeString, got {family:?}"
        );
        assert_eq!(somevalue_to_valuetype(&family), ValueType::Str);
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
