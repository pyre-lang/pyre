//! RPython `rpython/annotator/binaryop.py` — pairwise dispatch for
//! `dispatch=2` operations (`add` / `sub` / `mul` / ... / `is_` /
//! `getitem` / ...).
//!
//! Upstream this file is ~860 LOC of `@op.<name>.register(Some_cls_1,
//! Some_cls_2)` decorators. The Rust port fans the handlers out across
//! Commits 3-5 (one commit per pair-family) and registers them via
//! [`init`] — the `DoubleDispatchMixin._registry` / `_cache` population
//! upstream does at module-import time.
//!
//! Commit 3 (primitive arith) fills in:
//!
//! * `is__default` for `(SomeObject, SomeObject)` (binaryop.py)
//! * `pairtype(SomeObject, SomeObject)` fallback table (90-144)
//! * `pairtype(SomeInteger, SomeInteger)` arith (175-242)

use std::collections::HashMap;
use std::rc::Rc;

#[cfg(test)]
use super::super::flowspace::model::Variable;
use super::super::flowspace::model::{ConstValue, Constant, Hlvalue};
use super::super::flowspace::operation::{
    BuiltinException, CanOnlyThrow, HLOperation, OpKind, Specialization, Transformation, pure,
};
use super::annrpython::RPythonAnnotator;
use super::model::{
    self, KnownType, SomeBool, SomeByteArray, SomeChar, SomeFloat, SomeInteger, SomeObjectTrait,
    SomeString, SomeTuple, SomeUnicodeCodePoint, SomeUnicodeString, SomeValue, SomeValueTag,
    s_impossible_value, unionof,
};
use crate::tool::pairtype::DoubleDispatchRegistry;
use crate::translator::rtyper::llannotation;

/// RPython `BINARY_OPERATIONS` (binaryop.py).
///
/// ```python
/// BINARY_OPERATIONS = set([oper.opname for oper in op.__dict__.values()
///                         if oper.dispatch == 2])
/// ```
pub static BINARY_OPERATIONS: &[OpKind] = &[
    OpKind::Is,
    OpKind::GetItem,
    OpKind::GetItemIdx,
    OpKind::SetItem,
    OpKind::DelItem,
    OpKind::Add,
    OpKind::AddOvf,
    OpKind::Sub,
    OpKind::SubOvf,
    OpKind::Mul,
    OpKind::MulOvf,
    OpKind::TrueDiv,
    OpKind::FloorDiv,
    OpKind::FloorDivOvf,
    OpKind::Div,
    OpKind::DivOvf,
    OpKind::Mod,
    OpKind::ModOvf,
    OpKind::LShift,
    OpKind::LShiftOvf,
    OpKind::RShift,
    OpKind::And,
    OpKind::Or,
    OpKind::Xor,
    OpKind::InplaceAdd,
    OpKind::InplaceSub,
    OpKind::InplaceMul,
    OpKind::InplaceTrueDiv,
    OpKind::InplaceFloorDiv,
    OpKind::InplaceDiv,
    OpKind::InplaceMod,
    OpKind::InplaceLShift,
    OpKind::InplaceRShift,
    OpKind::InplaceAnd,
    OpKind::InplaceOr,
    OpKind::InplaceXor,
    OpKind::Lt,
    OpKind::Le,
    OpKind::Eq,
    OpKind::Ne,
    OpKind::Gt,
    OpKind::Ge,
    OpKind::Cmp,
    OpKind::Coerce,
];

/// Module-import time side effect matching upstream
/// `@op.<name>.register(...)` decorators in binaryop.py. Called from
/// the `_REGISTRY_DOUBLE` LazyLock initializer in `flowspace/operation.rs`.
///
/// The Rust port groups registrations by pair-family so each upstream
/// `class __extend__(pairtype(X, Y)):` block maps to a contiguous
/// section below.
pub(crate) fn init(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    init_is_default(reg);
    init_cmp_default(reg);
    init_object_pairtype(reg);
    init_integer_pairtype(reg);
    init_cmp_integer(reg);
    init_bool_pairtype(reg);
    init_float_pairtype(reg);
    init_singlefloat_pairtype(reg);
    init_longfloat_pairtype(reg);
    init_string_pairtype(reg);
    init_bytearray_pairtype(reg);
    init_bytearray_integer_pairtype(reg);
    init_string_bytearray_cross_pairtype(reg);
    init_char_pairtype(reg);
    init_unicodecp_pairtype(reg);
    init_string_unicodestring_mod(reg);
    init_string_tuple_mod(reg);
    init_string_object_mod(reg);
    init_list_list_pairtype(reg);
    init_tuple_tuple_pairtype(reg);
    init_tuple_integer_pairtype(reg);
    init_list_integer_pairtype(reg);
    init_rustvec_integer_pairtype(reg);
    init_string_integer_pairtype(reg);
    init_unicodestring_integer_pairtype(reg);
    init_integer_string_pairtype(reg);
    init_unicode_family_union_add(reg);
    init_cmp_str_unicode(reg);
    init_integer_list_pairtype(reg);
    llannotation::init_pairtypes(reg);
    init_dict_dict_pairtype(reg);
    init_dict_object_pairtype(reg);
    init_dict_getitem(reg);
    init_iterator_pairtype(reg);
    init_builtinmethod_pairtype(reg);
    init_pbc_pbc_is_(reg);
    init_impossible_none_pairtype(reg);
    init_pbc_object_pairtype(reg);
    init_none_object_pairtype(reg);
    init_pbc_string_pairtype(reg);
    init_none_string_pairtype(reg);
}

/// Module-import time side effect mirroring
/// `@op.<name>.register_transform(...)` decorators in binaryop.py. Runs
/// at `_TRANSFORM_DOUBLE` thread_local initialization; each block below
/// matches a single upstream `@op.X.register_transform(Y, Z)` decorator.
pub(crate) fn init_transform(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Transformation>>,
) {
    init_instance_object_transform(reg);
}

fn register_transform(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Transformation>>,
    op: OpKind,
    tag1: SomeValueTag,
    tag2: SomeValueTag,
    tx: Transformation,
) {
    reg.entry(op).or_default().set((tag1, tag2), tx);
}

fn register(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
    op: OpKind,
    tag1: SomeValueTag,
    tag2: SomeValueTag,
    spec: Specialization,
) {
    reg.entry(op).or_default().set((tag1, tag2), spec);
}

// binaryop.py:26-60 — @op.is_.register(SomeObject, SomeObject)

fn init_is_default(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::Is,
        SomeValueTag::Object,
        SomeValueTag::Object,
        Specialization {
            apply: pure(is__default),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

/// RPython `is__default(annotator, obj1, obj2)` (binaryop.py).
///
/// ```python
/// @op.is_.register(SomeObject, SomeObject)
/// def is__default(annotator, obj1, obj2):
///     r = SomeBool()
///     s_obj1 = annotator.annotation(obj1)
///     s_obj2 = annotator.annotation(obj2)
///     if s_obj2.is_constant():
///         if s_obj1.is_constant():
///             r.const = s_obj1.const is s_obj2.const
///         if s_obj2.const is None and not s_obj1.can_be_none():
///             r.const = False
///     elif s_obj1.is_constant():
///         if s_obj1.const is None and not s_obj2.can_be_none():
///             r.const = False
///     knowntypedata = defaultdict(dict)
///     bk = annotator.bookkeeper
///     def bind(src_obj, tgt_obj):
///         s_src = annotator.annotation(src_obj)
///         s_tgt = annotator.annotation(tgt_obj)
///         if hasattr(s_tgt, 'is_type_of') and s_src.is_constant():
///             add_knowntypedata(
///                 knowntypedata, True,
///                 s_tgt.is_type_of,
///                 bk.valueoftype(s_src.const))
///         add_knowntypedata(knowntypedata, True, [tgt_obj], s_src)
///         s_nonnone = s_tgt
///         if (s_src.is_constant() and s_src.const is None and
///                 s_tgt.can_be_none()):
///             s_nonnone = s_tgt.nonnoneify()
///         add_knowntypedata(knowntypedata, False, [tgt_obj], s_nonnone)
///     bind(obj2, obj1)
///     bind(obj1, obj2)
///     r.set_knowntypedata(knowntypedata)
///     return r
/// ```
#[allow(non_snake_case)]
pub fn is__default(annotator: &RPythonAnnotator, hlop: &HLOperation) -> SomeValue {
    let obj1 = &hlop.args[0];
    let obj2 = &hlop.args[1];
    let s_obj1 = annotator.annotation(obj1).unwrap_or(SomeValue::Impossible);
    let s_obj2 = annotator.annotation(obj2).unwrap_or(SomeValue::Impossible);

    let mut r = SomeBool::new();

    // binaryop.py:31-38.
    if s_obj2.is_constant() {
        if s_obj1.is_constant() {
            // Upstream: `r.const = s_obj1.const is s_obj2.const` — Python
            // object identity. For primitives (int/float/str/bool/None)
            // structural equality on `ConstValue` reproduces the Python
            // intern semantics.
            let eq = s_obj1.const_() == s_obj2.const_();
            r.base.const_box = Some(Constant::new(ConstValue::Bool(eq)));
        }
        if matches!(s_obj2.const_(), Some(ConstValue::None)) && !s_obj1.can_be_none() {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        }
    } else if s_obj1.is_constant()
        && matches!(s_obj1.const_(), Some(ConstValue::None))
        && !s_obj2.can_be_none()
    {
        r.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
    }

    // binaryop.py:39-59.
    let mut knowntypedata: model::KnownTypeData = HashMap::new();
    let bk = Rc::clone(&annotator.bookkeeper); // upstream: bk = annotator.bookkeeper

    // Upstream `def bind(src_obj, tgt_obj):` is a closure over
    // `knowntypedata`, `bk`, `annotator`. Rust inlines the two call
    // sites (bind(obj2, obj1); bind(obj1, obj2)) with explicit
    // parameters.
    bind_is(&mut knowntypedata, &bk, obj1, &s_obj2, &s_obj1);
    bind_is(&mut knowntypedata, &bk, obj2, &s_obj1, &s_obj2);
    r.set_knowntypedata(knowntypedata);
    SomeValue::Bool(r)
}

/// RPython `_annotation_key(t)` + `Bookkeeper.valueoftype` bridge.
///
/// Maps a constant's HostObject payload back onto an
/// [`super::signature::AnnotationSpec`] so the bookkeeper's type-to-
/// annotation lookup (`bk.valueoftype(t)`) can fire on the `is__default`
/// `type(x) is T` refinement path (binaryop.py).
///
/// Mirrors upstream's `_annotation_key` (signature.py) name-based
/// dispatch: builtin primitive types dispatch by qualname, everything
/// else (user classes, exception types) routes through
/// `AnnotationSpec::UserClass`.
fn host_to_annotation_spec(cv: &ConstValue) -> Option<super::signature::AnnotationSpec> {
    use super::signature::AnnotationSpec;
    let obj = match cv {
        ConstValue::HostObject(obj) => obj,
        _ => return None,
    };
    match obj.qualname() {
        "bool" => Some(AnnotationSpec::Bool),
        "int" => Some(AnnotationSpec::Int),
        "float" => Some(AnnotationSpec::Float),
        "str" => Some(AnnotationSpec::Str),
        "unicode" => Some(AnnotationSpec::Unicode),
        "NoneType" | "types.NoneType" => Some(AnnotationSpec::NoneType),
        "type" => Some(AnnotationSpec::Type),
        _ if obj.is_class() => Some(AnnotationSpec::UserClass(obj.clone())),
        _ => None,
    }
}

/// Inlined body of the `bind(src_obj, tgt_obj)` closure
/// (binaryop.py). `tgt_obj`'s `Variable` identity is the
/// knowntypedata key; `s_src`/`s_tgt` carry the annotations.
fn bind_is(
    knowntypedata: &mut model::KnownTypeData,
    bk: &Rc<super::bookkeeper::Bookkeeper>,
    tgt_obj: &Hlvalue,
    s_src: &SomeValue,
    s_tgt: &SomeValue,
) {
    // upstream binaryop.py:44-49:
    //     if hasattr(s_tgt, 'is_type_of') and s_src.is_constant():
    //         add_knowntypedata(knowntypedata, True,
    //                           s_tgt.is_type_of,
    //                           bk.valueoftype(s_src.const))
    //
    // `bk.valueoftype(s_src.const)` takes a host-level type object and
    // returns its `SomeValue`. The Rust port routes through
    // `host_to_annotation_spec` (HostObject → AnnotationSpec) then
    // `bk.valueoftype(&spec)`. When the HostObject isn't a recognised
    // type (e.g. a non-class constant), we fall back to widening by
    // `s_src` — monotone-safe and matches the prior Rust behaviour.
    if let SomeValue::TypeOf(t) = s_tgt
        && s_src.is_constant()
        && !t.is_type_of.is_empty()
    {
        let refined = s_src
            .const_()
            .and_then(host_to_annotation_spec)
            .and_then(|spec| bk.valueoftype(&spec).ok())
            .unwrap_or_else(|| s_src.clone());
        let vars: Vec<Rc<super::super::flowspace::model::Variable>> =
            t.is_type_of.iter().map(Rc::clone).collect();
        super::model::add_knowntypedata(
            knowntypedata,
            super::model::ExitCaseKey::Bool(true),
            &vars,
            refined,
        );
    }

    // binaryop.py:50 — `add_knowntypedata(..., True, [tgt_obj], s_src)`.
    let tgt_var = match tgt_obj {
        Hlvalue::Variable(v) => Some(Rc::new(v.clone())),
        Hlvalue::Constant(_) => None,
    };
    if let Some(var) = tgt_var.as_ref() {
        super::model::add_knowntypedata(
            knowntypedata,
            super::model::ExitCaseKey::Bool(true),
            &[Rc::clone(var)],
            s_src.clone(),
        );

        // binaryop.py — False branch `nonnoneify`.
        let s_nonnone = if s_src.is_constant()
            && matches!(s_src.const_(), Some(ConstValue::None))
            && s_tgt.can_be_none()
        {
            s_tgt.nonnoneify()
        } else {
            s_tgt.clone()
        };
        super::model::add_knowntypedata(
            knowntypedata,
            super::model::ExitCaseKey::Bool(false),
            &[Rc::clone(var)],
            s_nonnone,
        );
    }
}

// binaryop.py __extend__ — pairtype(SomeObject, SomeObject) fallback

fn init_object_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    use OpKind::*;
    use SomeValueTag::Object;

    // binaryop.py inplace_add — `inplace_xxx((obj1, obj2)): return pair(obj1, obj2).xxx()`.
    // Twelve separate methods upstream; twelve separate Rust
    // functions below, wired up with their upstream `can_only_throw`
    // side-bands (empty by default; ZeroDivisionError for the division
    // family per binaryop.py:113-116).
    let zd = || CanOnlyThrow::List(vec![BuiltinException::ZeroDivisionError]);
    let empty = || CanOnlyThrow::List(vec![]);
    register(
        reg,
        InplaceAdd,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_add),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceSub,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_sub),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceMul,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_mul),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceTrueDiv,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_truediv),
            can_only_throw: zd(),
        },
    );
    register(
        reg,
        InplaceFloorDiv,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_floordiv),
            can_only_throw: zd(),
        },
    );
    register(
        reg,
        InplaceDiv,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_div),
            can_only_throw: zd(),
        },
    );
    register(
        reg,
        InplaceMod,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_mod),
            can_only_throw: zd(),
        },
    );
    register(
        reg,
        InplaceLShift,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_lshift),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceRShift,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_rshift),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceAnd,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_and_),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceOr,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_or_),
            can_only_throw: empty(),
        },
    );
    register(
        reg,
        InplaceXor,
        Object,
        Object,
        Specialization {
            apply: pure(inplace_xor),
            can_only_throw: empty(),
        },
    );

    // binaryop.py — `cmp` default (immutable-const fold handled
    // by flowspace pyfunc; default returns SomeInteger).
    register(
        reg,
        GetItem,
        Object,
        Object,
        Specialization {
            apply: pure(getitem_default),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        GetItemIdx,
        Object,
        Object,
        Specialization {
            apply: pure(getitem_idx),
            can_only_throw: CanOnlyThrow::Callable(Box::new(getitem_can_only_throw)),
        },
    );
    register(
        reg,
        Cmp,
        Object,
        Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );

    // binaryop.py — `divmod((obj1, obj2))`.
    register(
        reg,
        DivMod,
        Object,
        Object,
        Specialization {
            apply: pure(|ann, hl| {
                // Upstream: `SomeTuple([pair(obj1, obj2).div(), pair(obj1, obj2).mod()])`.
                // Use explicit HLOperation dispatches via _REGISTRY_DOUBLE.
                let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
                let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
                let sv_div = dispatch_pair(OpKind::Div, ann, hl, s0.tag(), s1.tag());
                let sv_mod = dispatch_pair(OpKind::Mod, ann, hl, s0.tag(), s1.tag());
                SomeValue::Tuple(SomeTuple::new(vec![sv_div, sv_mod]))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );

    // binaryop.py — `coerce((obj1, obj2))` delegates to `pair.union()`.
    register(
        reg,
        Coerce,
        Object,
        Object,
        Specialization {
            apply: pure(|ann, hl| {
                let s1 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
                let s2 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
                super::model::union(&s1, &s2).unwrap_or(SomeValue::Impossible)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );

    // binaryop.py — `add = sub = mul = truediv = floordiv = div = mod =
    //   lshift = rshift = and_ = or_ = xor = delitem = s_ImpossibleValue`.
    for op in &[
        Add, Sub, Mul, TrueDiv, FloorDiv, Div, Mod, LShift, RShift, And, Or, Xor, DelItem,
    ] {
        register(
            reg,
            *op,
            Object,
            Object,
            Specialization {
                apply: pure(|_ann, _hl| SomeValue::Impossible),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }

    // binaryop.py — `def setitem((obj1, obj2), _): return s_ImpossibleValue`.
    register(
        reg,
        SetItem,
        Object,
        Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Impossible),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

/// RPython `getitem_default(ann, v_obj, v_index)` (binaryop.py).
pub fn getitem_default(_ann: &RPythonAnnotator, _hl: &HLOperation) -> SomeValue {
    s_impossible_value()
}

fn getitem_can_only_throw(args_s: &[SomeValue]) -> Option<Vec<BuiltinException>> {
    let [s_c1, s_o2, ..] = args_s else {
        return None;
    };
    Some(read_can_only_throw_for_pair(
        OpKind::GetItem,
        s_c1.tag(),
        s_o2.tag(),
        args_s,
    ))
}

/// RPython `getitem_idx(ann, v_obj, v_index)` (binaryop.py).
pub fn getitem_idx(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_obj = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s_index = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::GetItem, ann, hl, s_obj.tag(), s_index.tag())
}

// RPython `pairtype(SomeObject, SomeObject)` inplace_* methods
// (binaryop.py inplace_add). Each upstream method is a one-liner
// delegating to the non-inplace twin via `pair(obj1, obj2).xxx()`.
// Rust replays the same delegation through `dispatch_pair`, which
// looks up the `(OpKind, tag1, tag2)` entry in `_REGISTRY_DOUBLE`.

fn inplace_add(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Add, ann, hl, s0.tag(), s1.tag())
}

fn inplace_sub(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Sub, ann, hl, s0.tag(), s1.tag())
}

fn inplace_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Mul, ann, hl, s0.tag(), s1.tag())
}

fn inplace_truediv(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::TrueDiv, ann, hl, s0.tag(), s1.tag())
}

fn inplace_floordiv(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::FloorDiv, ann, hl, s0.tag(), s1.tag())
}

fn inplace_div(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Div, ann, hl, s0.tag(), s1.tag())
}

fn inplace_mod(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Mod, ann, hl, s0.tag(), s1.tag())
}

fn inplace_lshift(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::LShift, ann, hl, s0.tag(), s1.tag())
}

fn inplace_rshift(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::RShift, ann, hl, s0.tag(), s1.tag())
}

// Trailing underscore in `inplace_and_` / `inplace_or_` mirrors
// upstream's `inplace_and` / `inplace_or` — Python doesn't treat
// `and`/`or` as reserved in method name position, so upstream spells
// them without underscore; Rust reserves `and`/`or` as prefix-paths,
// so we use a trailing underscore to avoid clashes in readers' minds.
fn inplace_and_(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::And, ann, hl, s0.tag(), s1.tag())
}

fn inplace_or_(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Or, ann, hl, s0.tag(), s1.tag())
}

fn inplace_xor(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    dispatch_pair(OpKind::Xor, ann, hl, s0.tag(), s1.tag())
}

/// Helper for `pair(obj1, obj2).xxx()` — look up `(tag1, tag2)` in the
/// `_REGISTRY_DOUBLE[target]` pair registry and invoke the spec.
fn dispatch_pair(
    target: OpKind,
    ann: &RPythonAnnotator,
    hl: &HLOperation,
    tag1: SomeValueTag,
    tag2: SomeValueTag,
) -> SomeValue {
    crate::flowspace::operation::_REGISTRY_DOUBLE.with(|cell| {
        let reg = cell.borrow();
        let entries = reg.get(&target).expect("target registry missing");
        match entries.get((tag1, tag2), tag1.mro(), tag2.mro()) {
            // The delegated pair op is always a value-producing arithmetic
            // spec (never a void `setattr`-style handler), so its
            // `Optional[SomeObject]` result carries a concrete annotation.
            Some(spec) => crate::flowspace::operation::apply_specialization(spec, ann, hl)
                .unwrap_or_else(|err| std::panic::panic_any(err))
                .expect("delegated pair op is not a void operation"),
            None => SomeValue::Impossible,
        }
    })
}

fn read_can_only_throw_for_pair(
    op: OpKind,
    lhs: SomeValueTag,
    rhs: SomeValueTag,
    args_s: &[SomeValue],
) -> Vec<BuiltinException> {
    let mut result = op.canraise().to_vec();
    crate::flowspace::operation::_REGISTRY_DOUBLE.with(|cell| {
        if let Some(registry) = cell.borrow().get(&op)
            && let Some(spec) = registry.get((lhs, rhs), lhs.mro(), rhs.mro())
        {
            result = model::read_can_only_throw(&spec.can_only_throw, args_s)
                .unwrap_or_else(|| op.canraise().to_vec());
        }
    });
    result
}

// binaryop.py __extend__ — pairtype(SomeInteger, SomeInteger)

fn init_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    use OpKind::*;
    use SomeValueTag::Integer;

    // binaryop.py — `or_ = xor = add = mul = _clone(union, [])`.
    for op in &[Or, Xor, Add, Mul] {
        register(
            reg,
            *op,
            Integer,
            Integer,
            Specialization {
                apply: pure(integer_union),
                can_only_throw: CanOnlyThrow::List(vec![]),
            },
        );
    }
    // binaryop.py — `add_ovf = mul_ovf = _clone(union, [OverflowError])`.
    for op in &[AddOvf, MulOvf] {
        register(
            reg,
            *op,
            Integer,
            Integer,
            Specialization {
                apply: pure(integer_union),
                can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
            },
        );
    }
    // binaryop.py — `div = floordiv = mod = _clone(union, [ZeroDivisionError])`.
    for op in &[Div, FloorDiv, Mod] {
        register(
            reg,
            *op,
            Integer,
            Integer,
            Specialization {
                apply: pure(integer_union),
                can_only_throw: CanOnlyThrow::List(vec![BuiltinException::ZeroDivisionError]),
            },
        );
    }
    // binaryop.py:207 — `div_ovf = floordiv_ovf = mod_ovf =
    //                    _clone(union, [ZeroDivisionError, OverflowError])`.
    for op in &[DivOvf, FloorDivOvf, ModOvf] {
        register(
            reg,
            *op,
            Integer,
            Integer,
            Specialization {
                apply: pure(integer_union),
                can_only_throw: CanOnlyThrow::List(vec![
                    BuiltinException::ZeroDivisionError,
                    BuiltinException::OverflowError,
                ]),
            },
        );
    }

    // binaryop.py — `def truediv((int1, int2)): return SomeFloat()`.
    register(
        reg,
        TrueDiv,
        Integer,
        Integer,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::ZeroDivisionError]),
        },
    );

    // binaryop.py — `inplace_div = div; inplace_truediv = truediv`.
    register(
        reg,
        InplaceDiv,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_union),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::ZeroDivisionError]),
        },
    );
    register(
        reg,
        InplaceTrueDiv,
        Integer,
        Integer,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::ZeroDivisionError]),
        },
    );

    // binaryop.py — custom `sub` (restype without nonneg).
    register(
        reg,
        Sub,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_sub),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        SubOvf,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_sub),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );

    // binaryop.py — `and_` propagates nonneg.
    register(
        reg,
        And,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_and),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );

    // binaryop.py — `lshift`.
    register(
        reg,
        LShift,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_lshift),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        LShiftOvf,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_lshift),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );

    // binaryop.py — `rshift`.
    register(
        reg,
        RShift,
        Integer,
        Integer,
        Specialization {
            apply: pure(integer_rshift),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

/// RPython `pairtype(SomeInteger, SomeInteger).union((int1, int2))`
/// (binaryop.py). Returns the widening union of two integer
/// annotations. `_clone(union, …)` arith entries reuse this body.
///
/// Upstream raises `UnionError` when the signedness cannot be proven;
/// the Rust port surfaces that via `panic!` — matching the upstream
/// `raise` at binaryop.py/196/200.
fn integer_union(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    match super::model::union(&s0, &s1) {
        Ok(sv) => sv,
        Err(e) => panic!("UnionError: {}", e),
    }
}

/// RPython `pairtype(SomeInteger, SomeInteger).sub((int1, int2))`
/// (binaryop.py).
fn integer_sub(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    let kt = crate::rlib::rarithmetic::compute_restype(s0.knowntype(), s1.knowntype());
    SomeValue::Integer(SomeInteger::new_with_knowntype(false, kt))
}

/// RPython `pairtype(SomeInteger, SomeInteger).and_((int1, int2))`
/// (binaryop.py).
fn integer_and(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    let s1 = ann.annotation(&hl.args[1]).unwrap_or(SomeValue::Impossible);
    let nonneg = match (&s0, &s1) {
        (SomeValue::Integer(i0), SomeValue::Integer(i1)) => i0.nonneg || i1.nonneg,
        _ => false,
    };
    let kt = crate::rlib::rarithmetic::compute_restype(s0.knowntype(), s1.knowntype());
    SomeValue::Integer(SomeInteger::new_with_knowntype(nonneg, kt))
}

/// RPython `pairtype(SomeInteger, SomeInteger).lshift((int1, int2))`
/// (binaryop.py).
fn integer_lshift(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    match &s0 {
        SomeValue::Bool(_) => SomeValue::Integer(SomeInteger::default()),
        SomeValue::Integer(i) => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(false, i.knowntype()))
        }
        _ => SomeValue::Integer(SomeInteger::default()),
    }
}

/// RPython `pairtype(SomeInteger, SomeInteger).rshift((int1, int2))`
/// (binaryop.py).
fn integer_rshift(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s0 = ann.annotation(&hl.args[0]).unwrap_or(SomeValue::Impossible);
    match &s0 {
        SomeValue::Bool(_) => SomeValue::Integer(SomeInteger::new(true, false)),
        SomeValue::Integer(i) => {
            SomeValue::Integer(SomeInteger::new_with_knowntype(i.nonneg, i.knowntype()))
        }
        _ => SomeValue::Integer(SomeInteger::default()),
    }
}

// binaryop.py — _make_cmp_annotator_default

fn init_cmp_default(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream:
    //   def _make_cmp_annotator_default(cmp_op):
    //       @cmp_op.register(SomeObject, SomeObject)
    //       def default_annotate(annotator, obj1, obj2):
    //           ...
    //   for cmp_op in [op.lt, op.le, op.eq, op.ne, op.gt, op.ge]:
    //       _make_cmp_annotator_default(cmp_op)
    for op in &[
        OpKind::Lt,
        OpKind::Le,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Gt,
        OpKind::Ge,
    ] {
        let cmp_op = *op;
        register(
            reg,
            cmp_op,
            SomeValueTag::Object,
            SomeValueTag::Object,
            Specialization {
                apply: pure(move |ann, hl| cmp_default_annotate(cmp_op, ann, hl)),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

/// RPython `default_annotate(annotator, obj1, obj2)` (binaryop.py).
///
/// ```python
/// def default_annotate(annotator, obj1, obj2):
///     s_1, s_2 = annotator.annotation(obj1), annotator.annotation(obj2)
///     if s_1.is_immutable_constant() and s_2.is_immutable_constant():
///         return immutablevalue(cmp_op.pyfunc(s_1.const, s_2.const))
///     else:
///         return s_Bool
/// ```
fn cmp_default_annotate(
    cmp_op: OpKind,
    annotator: &RPythonAnnotator,
    hl: &HLOperation,
) -> SomeValue {
    let s_1 = annotator
        .annotation(&hl.args[0])
        .unwrap_or(SomeValue::Impossible);
    let s_2 = annotator
        .annotation(&hl.args[1])
        .unwrap_or(SomeValue::Impossible);
    if s_1.is_immutable_constant() && s_2.is_immutable_constant() {
        let c1 = s_1
            .const_()
            .expect("is_immutable_constant implies const set");
        let c2 = s_2
            .const_()
            .expect("is_immutable_constant implies const set");
        if let Some(result) = crate::flowspace::operation::pyfunc(cmp_op, &[c1, c2])
            && let Ok(sv) = annotator.bookkeeper.immutablevalue(&result)
        {
            return sv;
        }
    }
    super::model::s_bool()
}

// binaryop.py — _make_cmp_annotator_int

fn init_cmp_integer(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    for op in &[
        OpKind::Lt,
        OpKind::Le,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Gt,
        OpKind::Ge,
    ] {
        let cmp_op = *op;
        register(
            reg,
            cmp_op,
            SomeValueTag::Integer,
            SomeValueTag::Integer,
            Specialization {
                apply: pure(move |ann, hl| cmp_integer(cmp_op, ann, hl)),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

/// RPython `_compare_helper(annotator, int1, int2)` (binaryop.py).
///
/// Branch-refinement cmp: the returned SomeBool carries knowntypedata
/// propagating `nonneg` information between the two operands for the
/// True / False branches.
fn cmp_integer(cmp_op: OpKind, annotator: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let obj1 = &hl.args[0];
    let obj2 = &hl.args[1];
    let s_int1 = annotator.annotation(obj1).unwrap_or(SomeValue::Impossible);
    let s_int2 = annotator.annotation(obj2).unwrap_or(SomeValue::Impossible);
    let mut r = SomeBool::new();
    // binaryop.py:250-251 — constant fold.
    if s_int1.is_immutable_constant() && s_int2.is_immutable_constant() {
        let c1 = s_int1
            .const_()
            .expect("is_immutable_constant implies const set");
        let c2 = s_int2
            .const_()
            .expect("is_immutable_constant implies const set");
        if let Some(result) = crate::flowspace::operation::pyfunc(cmp_op, &[c1, c2]) {
            r.base.const_box = Some(Constant::new(result));
        }
    }
    // binaryop.py:263-265 — nonneg deduction only when both sides signed.
    let kt1 = s_int1.knowntype();
    let kt2 = s_int2.knowntype();
    if !(crate::rlib::rarithmetic::signedtype(kt1) && crate::rlib::rarithmetic::signedtype(kt2)) {
        return SomeValue::Bool(r);
    }
    // binaryop.py:266-278 — build knowntypedata.
    let mut knowntypedata: model::KnownTypeData = HashMap::new();
    let tointtype = |kt: KnownType| {
        if kt == KnownType::Bool {
            KnownType::Int
        } else {
            kt
        }
    };

    // binaryop.py:271-274 — `if s_int1.nonneg and isinstance(int2, Variable):
    //                        case = cmp_op.opname in ('lt', 'le', 'eq')
    //                        add_knowntypedata(..., case, [int2], SomeInteger(nonneg=True, knowntype=tointtype(s_int2)))`.
    if let SomeValue::Integer(i1) = &s_int1
        && i1.nonneg
        && let Hlvalue::Variable(v) = obj2
    {
        let case = matches!(cmp_op, OpKind::Lt | OpKind::Le | OpKind::Eq);
        super::model::add_knowntypedata(
            &mut knowntypedata,
            super::model::ExitCaseKey::Bool(case),
            &[Rc::new(v.clone())],
            SomeValue::Integer(SomeInteger::new_with_knowntype(true, tointtype(kt2))),
        );
    }
    // binaryop.py:275-278 — symmetric.
    if let SomeValue::Integer(i2) = &s_int2
        && i2.nonneg
        && let Hlvalue::Variable(v) = obj1
    {
        let case = matches!(cmp_op, OpKind::Gt | OpKind::Ge | OpKind::Eq);
        super::model::add_knowntypedata(
            &mut knowntypedata,
            super::model::ExitCaseKey::Bool(case),
            &[Rc::new(v.clone())],
            SomeValue::Integer(SomeInteger::new_with_knowntype(true, tointtype(kt1))),
        );
    }
    r.set_knowntypedata(knowntypedata);

    // binaryop.py — special case `x < 0` / `x >= 0` when `int2`
    // is a flow-graph Constant.
    if let Hlvalue::Constant(c2) = obj2
        && matches!(&c2.value, ConstValue::Int(0))
        && let SomeValue::Integer(i1) = &s_int1
        && i1.nonneg
    {
        if cmp_op == OpKind::Lt {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        } else if cmp_op == OpKind::Ge {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
        }
    }
    SomeValue::Bool(r)
}

// binaryop.py __extend__ — pairtype(SomeBool, SomeBool)

fn init_bool_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    use OpKind::*;
    use SomeValueTag::Bool;
    register(
        reg,
        And,
        Bool,
        Bool,
        Specialization {
            apply: pure(bool_and_),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        Or,
        Bool,
        Bool,
        Specialization {
            apply: pure(bool_or_),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        Xor,
        Bool,
        Bool,
        Specialization {
            apply: pure(bool_xor),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

/// RPython `pairtype(SomeBool, SomeBool).and_((boo1, boo2))`
/// (binaryop.py).
fn bool_and_(annotator: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = annotator
        .annotation(&hl.args[0])
        .unwrap_or(SomeValue::Impossible);
    let s2 = annotator
        .annotation(&hl.args[1])
        .unwrap_or(SomeValue::Impossible);
    let mut s = SomeBool::new();
    if s1.is_constant() {
        if matches!(s1.const_(), Some(ConstValue::Bool(false))) {
            s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        } else {
            return s2;
        }
    }
    if s2.is_constant() && matches!(s2.const_(), Some(ConstValue::Bool(false))) {
        s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
    }
    SomeValue::Bool(s)
}

/// RPython `pairtype(SomeBool, SomeBool).or_((boo1, boo2))`
/// (binaryop.py).
fn bool_or_(annotator: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = annotator
        .annotation(&hl.args[0])
        .unwrap_or(SomeValue::Impossible);
    let s2 = annotator
        .annotation(&hl.args[1])
        .unwrap_or(SomeValue::Impossible);
    let mut s = SomeBool::new();
    if s1.is_constant() {
        if matches!(s1.const_(), Some(ConstValue::Bool(true))) {
            s.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
        } else {
            return s2;
        }
    }
    if s2.is_constant() && matches!(s2.const_(), Some(ConstValue::Bool(true))) {
        s.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
    }
    SomeValue::Bool(s)
}

/// RPython `pairtype(SomeBool, SomeBool).xor((boo1, boo2))`
/// (binaryop.py).
fn bool_xor(annotator: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = annotator
        .annotation(&hl.args[0])
        .unwrap_or(SomeValue::Impossible);
    let s2 = annotator
        .annotation(&hl.args[1])
        .unwrap_or(SomeValue::Impossible);
    let mut s = SomeBool::new();
    if s1.is_constant()
        && s2.is_constant()
        && let (Some(ConstValue::Bool(b1)), Some(ConstValue::Bool(b2))) = (s1.const_(), s2.const_())
    {
        s.base.const_box = Some(Constant::new(ConstValue::Bool(b1 ^ b2)));
    }
    SomeValue::Bool(s)
}

// binaryop.py __extend__ — pairtype(SomeFloat, SomeFloat)

fn init_float_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    use OpKind::*;
    use SomeValueTag::Float;
    // binaryop.py — `add = sub = mul = union`.
    for op in &[Add, Sub, Mul] {
        register(
            reg,
            *op,
            Float,
            Float,
            Specialization {
                apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // binaryop.py — `div` + `truediv = div`, `can_only_throw = []`.
    register(
        reg,
        Div,
        Float,
        Float,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        TrueDiv,
        Float,
        Float,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // binaryop.py — `inplace_div = div; inplace_truediv = truediv`.
    register(
        reg,
        InplaceDiv,
        Float,
        Float,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        InplaceTrueDiv,
        Float,
        Float,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

// binaryop.py __extend__ — pairtype(SomeSingleFloat/LongFloat, ...)

fn init_singlefloat_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream class body only defines `union`, which is handled by
    // the union dispatch (annotator/model.rs) — no binary operation
    // registrations.
}

fn init_longfloat_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream: `class __extend__(pairtype(SomeLongFloat, SomeLongFloat))`
    // only defines `union`. Same note as `init_singlefloat_pairtype`.
}

// binaryop.py __extend__ — pairtype(SomeString, SomeString)

fn init_string_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `add`
    //
    //     def add((str1, str2)):
    //         # propagate const-ness to help getattr(obj, 'prefix' + const_name)
    //         result = SomeString(no_nul=str1.no_nul and str2.no_nul)
    //         if str1.is_immutable_constant() and str2.is_immutable_constant():
    //             result.const = str1.const + str2.const
    //         return result
    register(
        reg,
        OpKind::Add,
        SomeValueTag::String,
        SomeValueTag::String,
        Specialization {
            apply: pure(string_string_add),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // Note: `union` is handled in `annotator::model::union()` (already
    // covers (SomeString, SomeString)) — nothing to register here.
}

fn string_string_add(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    // `SomeChar` is a `SomeString` subclass (model.py), and
    // `pairtype(SomeChar, SomeChar)` (binaryop.py) registers only
    // `union`, so a char operand on either side resolves to this arm
    // through the MRO and the result is a `SomeString` either way.
    fn extract(sv: &SomeValue, side: &str) -> (bool, Option<Vec<u8>>) {
        let (inner, is_immutable_constant) = match sv {
            SomeValue::String(s) => (&s.inner, s.is_immutable_constant()),
            SomeValue::Char(c) => (&c.inner, c.is_immutable_constant()),
            other => panic!("string_string_add: {side} is not SomeString/SomeChar: {other:?}"),
        };
        // upstream reads `str.const` only under `is_immutable_constant()`,
        // so a non-constant operand contributes no const half.
        let bytes = is_immutable_constant
            .then(|| match inner.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::ByteStr(b)) => Some(b.clone()),
                _ => None,
            })
            .flatten();
        (inner.no_nul, bytes)
    }
    let a = ann
        .annotation(&hl.args[0])
        .expect("string_string_add: lhs unbound");
    let b = ann
        .annotation(&hl.args[1])
        .expect("string_string_add: rhs unbound");
    let (no_nul1, c1) = extract(&a, "arg 0");
    let (no_nul2, c2) = extract(&b, "arg 1");
    let mut result = SomeString::new(false, no_nul1 && no_nul2);
    // upstream: `if str1.is_immutable_constant() and str2.is_immutable_constant():
    //              result.const = str1.const + str2.const`
    if let (Some(a), Some(b)) = (c1, c2) {
        let mut combined = a;
        combined.extend_from_slice(&b);
        result.inner.base.const_box = Some(Constant::new(ConstValue::ByteStr(combined)));
    }
    SomeValue::String(result)
}

// binaryop.py __extend__ — pairtype(SomeByteArray, SomeByteArray)

fn init_bytearray_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `add`
    //
    //     def add((b1, b2)):
    //         return SomeByteArray()
    register(
        reg,
        OpKind::Add,
        SomeValueTag::ByteArray,
        SomeValueTag::ByteArray,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::ByteArray(SomeByteArray::new(false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // `union` handled in `model::union()` (not present there yet for
    // SomeByteArray — follow-up union wiring lives with the type-
    // lattice commits, not this dispatch commit).
}

// binaryop.py __extend__ — pairtype(SomeByteArray, SomeInteger)

fn init_bytearray_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `getitem((s_b, s_i))`: returns SomeInteger()
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::ByteArray,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // binaryop.py — `setitem((s_b, s_i), s_i2)`: asserts SomeInteger, no return
    //
    //     def setitem((s_b, s_i), s_i2):
    //         assert isinstance(s_i2, SomeInteger)
    //
    // The third operand (s_i2 = value) comes in hl.args[2]. Upstream
    // `setitem` specializations take (s_b, s_i) and an `s_value`
    // third-positional; the Rust port collapses all three into
    // `hl.args` and consumes the value via `ann.annotation`.
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::ByteArray,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(bytearray_integer_setitem),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn bytearray_integer_setitem(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    // upstream asserts the 3rd arg is a SomeInteger.
    let s_value = ann
        .annotation(&hl.args[2])
        .expect("bytearray.setitem: missing third-arg annotation");
    assert!(
        matches!(s_value, SomeValue::Integer(_)),
        "bytearray.setitem: assert isinstance(s_i2, SomeInteger): got {:?}",
        s_value
    );
    // upstream `setitem` returns None (void): consider binds the result
    // to Impossible without blocking.
    None
}

// binaryop.py — pairtype cross-str/bytearray for `add`
//
//     class __extend__(pairtype(SomeString, SomeByteArray),
//                      pairtype(SomeByteArray, SomeString),
//                      pairtype(SomeChar, SomeByteArray),
//                      pairtype(SomeByteArray, SomeChar)):
//         def add((b1, b2)):
//             return SomeByteArray()

fn init_string_bytearray_cross_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    let cross: &[(SomeValueTag, SomeValueTag)] = &[
        (SomeValueTag::String, SomeValueTag::ByteArray),
        (SomeValueTag::ByteArray, SomeValueTag::String),
        (SomeValueTag::Char, SomeValueTag::ByteArray),
        (SomeValueTag::ByteArray, SomeValueTag::Char),
    ];
    for (t1, t2) in cross {
        register(
            reg,
            OpKind::Add,
            *t1,
            *t2,
            Specialization {
                apply: pure(|_ann, _hl| SomeValue::ByteArray(SomeByteArray::new(false))),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// binaryop.py __extend__ — pairtype(SomeChar, SomeChar)

fn init_char_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream only defines `union`, already covered in model::union().
}

// binaryop.py __extend__ — pairtype(SomeUnicodeCodePoint, SomeUnicodeCodePoint)

fn init_unicodecp_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream only defines `union`, already covered in model::union().
}

// binaryop.py __extend__ — pairtype(SomeString, SomeUnicodeString) / reverse
//
//     class __extend__(pairtype(SomeString, SomeUnicodeString),
//                      pairtype(SomeUnicodeString, SomeString)):
//         def mod((str, unistring)):
//             raise AnnotatorError(
//                 "string formatting mixing strings and unicode not supported")

fn init_string_unicodestring_mod(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    for (t1, t2) in &[
        (SomeValueTag::String, SomeValueTag::UnicodeString),
        (SomeValueTag::UnicodeString, SomeValueTag::String),
    ] {
        register(
            reg,
            OpKind::Mod,
            *t1,
            *t2,
            Specialization {
                apply: pure(|_ann, _hl| {
                    panic!(
                        "AnnotatorError: string formatting mixing strings and unicode not supported"
                    )
                }),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// binaryop.py __extend__ — pairtype(SomeString, SomeTuple) /
//                       pairtype(SomeUnicodeString, SomeTuple)

fn init_string_tuple_mod(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream body (binaryop.py mod):
    //
    //     def mod((s_string, s_tuple)):
    //         if not s_string.is_constant():
    //             raise AnnotatorError("string formatting requires a constant "
    //                                  "string/unicode on the left of '%'")
    //         is_string   = isinstance(s_string, SomeString)
    //         is_unicode  = isinstance(s_string, SomeUnicodeString)
    //         assert is_string or is_unicode
    //         for s_item in s_tuple.items:
    //             if (is_unicode and isinstance(s_item, (SomeChar, SomeString)) or
    //                 is_string and isinstance(s_item, (SomeUnicodeCodePoint,
    //                                                   SomeUnicodeString))):
    //                 raise AnnotatorError("string formatting mixing strings and unicode not supported")
    //         no_nul = s_string.no_nul
    //         for s_item in s_tuple.items:
    //             if isinstance(s_item, SomeFloat):
    //                 pass
    //             elif (isinstance(s_item, SomeString) or
    //                   isinstance(s_item, SomeUnicodeString)) and s_item.no_nul:
    //                 pass
    //             else:
    //                 no_nul = False
    //                 break
    //         return s_string.__class__(no_nul=no_nul)
    for t1 in &[SomeValueTag::String, SomeValueTag::UnicodeString] {
        let t1_copy = *t1;
        register(
            reg,
            OpKind::Mod,
            t1_copy,
            SomeValueTag::Tuple,
            Specialization {
                apply: pure(move |ann, hl| string_tuple_mod(ann, hl, t1_copy)),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

fn string_tuple_mod(ann: &RPythonAnnotator, hl: &HLOperation, lhs_tag: SomeValueTag) -> SomeValue {
    let s_string = ann
        .annotation(&hl.args[0])
        .expect("string_tuple_mod: missing lhs annotation");
    let s_tuple = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::Tuple(t)) => t,
        other => panic!("string_tuple_mod: rhs not SomeTuple: {:?}", other),
    };
    if !s_string.is_constant() {
        panic!(
            "AnnotatorError: string formatting requires a constant string/unicode on the left of '%'"
        );
    }
    let is_string = matches!(lhs_tag, SomeValueTag::String);
    let is_unicode = matches!(lhs_tag, SomeValueTag::UnicodeString);
    assert!(is_string || is_unicode);
    for s_item in &s_tuple.items {
        let mixes = (is_unicode && matches!(s_item, SomeValue::Char(_) | SomeValue::String(_)))
            || (is_string
                && matches!(
                    s_item,
                    SomeValue::UnicodeCodePoint(_) | SomeValue::UnicodeString(_)
                ));
        if mixes {
            panic!("AnnotatorError: string formatting mixing strings and unicode not supported");
        }
    }
    let src_no_nul = match &s_string {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        _ => unreachable!("lhs_tag restricted to String / UnicodeString"),
    };
    let mut no_nul = src_no_nul;
    for s_item in &s_tuple.items {
        match s_item {
            SomeValue::Float(_) | SomeValue::Integer(_) | SomeValue::Bool(_) => {}
            SomeValue::String(s) if s.inner.no_nul => {}
            SomeValue::UnicodeString(s) if s.inner.no_nul => {}
            _ => {
                no_nul = false;
                break;
            }
        }
    }
    // upstream: `return s_string.__class__(no_nul=no_nul)` — same family class.
    match lhs_tag {
        SomeValueTag::String => SomeValue::String(SomeString::new(false, no_nul)),
        SomeValueTag::UnicodeString => {
            SomeValue::UnicodeString(SomeUnicodeString::new(false, no_nul))
        }
        _ => unreachable!("string_tuple_mod: lhs_tag restricted"),
    }
}

// binaryop.py __extend__ — pairtype(SomeString, SomeObject) /
//                       pairtype(SomeUnicodeString, SomeObject)

fn init_string_object_mod(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream body (binaryop.py mod):
    //
    //     def mod((s_string, s_arg)):
    //         assert not isinstance(s_arg, SomeTuple)
    //         return pair(s_string, SomeTuple([s_arg])).mod()
    for t1 in &[SomeValueTag::String, SomeValueTag::UnicodeString] {
        let t1_copy = *t1;
        register(
            reg,
            OpKind::Mod,
            t1_copy,
            SomeValueTag::Object,
            Specialization {
                apply: pure(move |ann, hl| string_object_mod(ann, hl, t1_copy)),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

fn string_object_mod(ann: &RPythonAnnotator, hl: &HLOperation, lhs_tag: SomeValueTag) -> SomeValue {
    let s_arg = ann
        .annotation(&hl.args[1])
        .expect("string_object_mod: missing rhs annotation");
    // upstream: `assert not isinstance(s_arg, SomeTuple)`.
    assert!(
        !matches!(s_arg, SomeValue::Tuple(_)),
        "string_object_mod: s_arg must not be SomeTuple"
    );
    // upstream: `return pair(s_string, SomeTuple([s_arg])).mod()` — delegate
    // to the (String|UnicodeString, Tuple) handler with a singleton tuple.
    let wrapped = SomeTuple::new(vec![s_arg]);
    // Synthesize a minimal HLOperation whose args[1] is a Constant carrying
    // the tuple annotation. We can't build a real Variable with an
    // annotation here without an annotator mutation, so the direct path is
    // to call string_tuple_mod with the synthesized tuple inline — bypass
    // the registry to keep the call parity-shaped.
    let s_string = ann
        .annotation(&hl.args[0])
        .expect("string_object_mod: missing lhs annotation");
    if !s_string.is_constant() {
        panic!(
            "AnnotatorError: string formatting requires a constant string/unicode on the left of '%'"
        );
    }
    let is_string = matches!(lhs_tag, SomeValueTag::String);
    let is_unicode = matches!(lhs_tag, SomeValueTag::UnicodeString);
    assert!(is_string || is_unicode);
    let src_no_nul = match &s_string {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        _ => unreachable!(),
    };
    let mut no_nul = src_no_nul;
    for s_item in &wrapped.items {
        match s_item {
            SomeValue::Float(_) | SomeValue::Integer(_) | SomeValue::Bool(_) => {}
            SomeValue::String(s) if s.inner.no_nul => {}
            SomeValue::UnicodeString(s) if s.inner.no_nul => {}
            _ => {
                no_nul = false;
                break;
            }
        }
    }
    match lhs_tag {
        SomeValueTag::String => SomeValue::String(SomeString::new(false, no_nul)),
        SomeValueTag::UnicodeString => {
            SomeValue::UnicodeString(SomeUnicodeString::new(false, no_nul))
        }
        _ => unreachable!(),
    }
}

// binaryop.py __extend__ — pairtype(SomeList, SomeList)

fn init_list_list_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `add((lst1, lst2)):`
    //
    //     def add((lst1, lst2)):
    //         bk = getbookkeeper()
    //         return lst1.listdef.offspring(bk, lst2.listdef)
    register(
        reg,
        OpKind::Add,
        SomeValueTag::List,
        SomeValueTag::List,
        Specialization {
            apply: pure(list_list_add),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // binaryop.py — `eq`, `ne = eq`: both call listdef.agree and return s_Bool.
    for op in &[OpKind::Eq, OpKind::Ne] {
        register(
            reg,
            *op,
            SomeValueTag::List,
            SomeValueTag::List,
            Specialization {
                apply: pure(list_list_eq),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // Note: `union` handled in `model::union()`.
}

fn list_list_add(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let lst1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_list_add: arg 0 is not SomeList"),
    };
    let lst2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_list_add: arg 1 is not SomeList"),
    };
    let s_new = lst1
        .listdef
        .offspring(&ann.bookkeeper, &[&lst2.listdef])
        .expect("list.offspring failed");
    SomeValue::List(s_new)
}

fn list_list_eq(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let lst1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_list_eq: arg 0 is not SomeList"),
    };
    let lst2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_list_eq: arg 1 is not SomeList"),
    };
    lst1.listdef
        .agree(&ann.bookkeeper, &lst2.listdef)
        .expect("listdef.agree failed");
    SomeValue::Bool(SomeBool::new())
}

// binaryop.py __extend__ — pairtype(SomeTuple, SomeTuple)

fn init_tuple_tuple_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `add((tup1, tup2))`:
    //     return SomeTuple(items = tup1.items + tup2.items)
    register(
        reg,
        OpKind::Add,
        SomeValueTag::Tuple,
        SomeValueTag::Tuple,
        Specialization {
            apply: pure(tuple_tuple_add),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // binaryop.py — `eq`, `ne = eq`: call .union() and return s_Bool.
    for op in &[OpKind::Eq, OpKind::Ne] {
        register(
            reg,
            *op,
            SomeValueTag::Tuple,
            SomeValueTag::Tuple,
            Specialization {
                apply: pure(tuple_tuple_eq),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // binaryop.py — `lt/le/gt/ge`: raise AnnotatorError.
    for op in &[OpKind::Lt, OpKind::Le, OpKind::Gt, OpKind::Ge] {
        let op_copy = *op;
        register(
            reg,
            op_copy,
            SomeValueTag::Tuple,
            SomeValueTag::Tuple,
            Specialization {
                apply: pure(move |_ann, _hl| {
                    let sym = match op_copy {
                        OpKind::Lt => "<",
                        OpKind::Le => "<=",
                        OpKind::Gt => ">",
                        OpKind::Ge => ">=",
                        _ => unreachable!(),
                    };
                    panic!("AnnotatorError: unsupported: (...) {sym} (...)")
                }),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

fn tuple_tuple_add(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let t1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::Tuple(t)) => t,
        _ => panic!("tuple_tuple_add: arg 0 not SomeTuple"),
    };
    let t2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::Tuple(t)) => t,
        _ => panic!("tuple_tuple_add: arg 1 not SomeTuple"),
    };
    let mut items = t1.items.clone();
    items.extend(t2.items.iter().cloned());
    SomeValue::Tuple(SomeTuple::new(items))
}

fn tuple_tuple_eq(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    // upstream: `tup1tup2.union(); return s_Bool` — side-effect union
    // call raises UnionError on length mismatch, which upstream wraps
    // as AnnotatorError via `complete()`. Rust port re-uses model::union.
    let t1 = ann.annotation(&hl.args[0]).expect("eq: lhs unbound");
    let t2 = ann.annotation(&hl.args[1]).expect("eq: rhs unbound");
    model::union(&t1, &t2).expect("tuple_tuple_eq: UnionError");
    SomeValue::Bool(SomeBool::new())
}

// binaryop.py __extend__ — pairtype(SomeTuple, SomeInteger)

fn init_tuple_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `getitem((tup1, int2))`:
    //
    //     def getitem((tup1, int2)):
    //         if int2.is_immutable_constant():
    //             try:
    //                 return tup1.items[int2.const]
    //             except IndexError:
    //                 return s_ImpossibleValue
    //         else:
    //             return unionof(*tup1.items)
    //     getitem.can_only_throw = [IndexError]
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::Tuple,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(tuple_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::IndexError]),
        },
    );
}

fn tuple_integer_getitem(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let tup1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::Tuple(t)) => t,
        _ => panic!("tuple_int_getitem: arg 0 not SomeTuple"),
    };
    let int2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::Integer(i)) => i,
        _ => panic!("tuple_int_getitem: arg 1 not SomeInteger"),
    };
    if int2.is_immutable_constant()
        && let Some(c) = &int2.base.const_box
        && let ConstValue::Int(idx) = c.value
    {
        let idx_usize = if idx < 0 {
            let adj = tup1.items.len() as i64 + idx;
            if adj < 0 {
                return s_impossible_value();
            }
            adj as usize
        } else {
            idx as usize
        };
        if idx_usize < tup1.items.len() {
            return tup1.items[idx_usize].clone();
        }
        return s_impossible_value();
    }
    unionof(tup1.items.iter()).expect("tuple_int_getitem: unionof failed")
}

// binaryop.py __extend__ — pairtype(SomeList, SomeInteger)

fn init_list_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `mul((lst1, int2))`: offspring(bk).
    register(
        reg,
        OpKind::Mul,
        SomeValueTag::List,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(list_integer_mul),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // binaryop.py — `getitem((lst1, int2))`: read_item. `can_only_throw = []`.
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::List,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(list_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // binaryop.py — `getitem_idx`: read_item, `can_only_throw = [IndexError]`.
    register(
        reg,
        OpKind::GetItemIdx,
        SomeValueTag::List,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(list_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::IndexError]),
        },
    );
    // binaryop.py — `setitem((lst1, int2), s_value)`: mutate + generalize.
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::List,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(list_integer_setitem),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::IndexError]),
        },
    );
    // binaryop.py — `delitem((lst1, int2))`: resize.
    register(
        reg,
        OpKind::DelItem,
        SomeValueTag::List,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(list_integer_delitem),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::IndexError]),
        },
    );
}

fn list_integer_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let lst1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_integer_mul: arg 0 not SomeList"),
    };
    // upstream: `lst1.listdef.offspring(bk)` — zero-other path.
    let s_new = lst1
        .listdef
        .offspring(&ann.bookkeeper, &[])
        .expect("offspring failed");
    SomeValue::List(s_new)
}

// `SomeRustVec` indexing: the item annotation is fixed by the Rust item
// type, so a read answers it and a write neither mutates nor generalizes.

fn init_rustvec_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::RustVec,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(rustvec_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::RustVec,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(|_ann, _hl| None),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

fn rustvec_integer_getitem(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    match ann.annotation(&hl.args[0]) {
        Some(SomeValue::RustVec(s)) => *s.s_item,
        _ => panic!("rustvec_integer_getitem: arg 0 not SomeRustVec"),
    }
}

fn list_integer_getitem(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let lst1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_integer_getitem: arg 0 not SomeList"),
    };
    // upstream: `position = getbookkeeper().position_key`. `None` is
    // valid here (no reflow frame active); `read_item` drops the
    // empty key from its read-locations set.
    let position = ann.bookkeeper.current_position_key();
    lst1.listdef.read_item(position)
}

fn list_integer_setitem(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    let lst1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_integer_setitem: arg 0 not SomeList"),
    };
    let s_value = ann
        .annotation(&hl.args[2])
        .expect("list_integer_setitem: missing value arg");
    lst1.listdef.mutate().expect("listdef.mutate failed");
    lst1.listdef
        .generalize(&s_value)
        .expect("listdef.generalize failed");
    // upstream `setitem` returns None (void).
    None
}

fn list_integer_delitem(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    let lst1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("list_integer_delitem: arg 0 not SomeList"),
    };
    // upstream binaryop.py delitem:
    //     def delitem((lst1, int2)):
    //         lst1.listdef.resize()
    lst1.listdef.resize().expect("listdef.resize failed");
    // upstream `delitem` returns None (void).
    None
}

// binaryop.py __extend__ — pairtype(SomeString, SomeInteger)

fn init_string_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `getitem((str1, int2))`: SomeChar(no_nul=str1.no_nul).
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::String,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(string_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::GetItemIdx,
        SomeValueTag::String,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(string_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::IndexError]),
        },
    );
    // binaryop.py — `mul((str1, int2))`: SomeString(no_nul=str1.no_nul).
    register(
        reg,
        OpKind::Mul,
        SomeValueTag::String,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(string_integer_mul),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn string_integer_getitem(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::String(s)) => s,
        _ => panic!("string_integer_getitem: arg 0 not SomeString"),
    };
    SomeValue::Char(SomeChar::new(s1.inner.no_nul))
}

fn string_integer_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::String(s)) => s,
        _ => panic!("string_integer_mul: arg 0 not SomeString"),
    };
    SomeValue::String(SomeString::new(false, s1.inner.no_nul))
}

// binaryop.py __extend__ — pairtype(SomeUnicodeString, SomeInteger)

fn init_unicodestring_integer_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::UnicodeString,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(unicodestring_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::GetItemIdx,
        SomeValueTag::UnicodeString,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(unicodestring_integer_getitem),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::IndexError]),
        },
    );
    register(
        reg,
        OpKind::Mul,
        SomeValueTag::UnicodeString,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(unicodestring_integer_mul),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn unicodestring_integer_getitem(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::UnicodeString(s)) => s,
        _ => panic!("unicodestring_integer_getitem: arg 0 not SomeUnicodeString"),
    };
    SomeValue::UnicodeCodePoint(SomeUnicodeCodePoint::new(s1.inner.no_nul))
}

fn unicodestring_integer_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::UnicodeString(s)) => s,
        _ => panic!("unicodestring_integer_mul: arg 0 not SomeUnicodeString"),
    };
    SomeValue::UnicodeString(SomeUnicodeString::new(false, s1.inner.no_nul))
}

// binaryop.py __extend__ — pairtype(SomeInteger, SomeString|SomeUnicodeString)

fn init_integer_string_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // upstream: `mul((int1, str2))`: `return str2.basestringclass(no_nul=str2.no_nul)`.
    //
    //     pairtype(SomeInteger, SomeString): -> SomeString(no_nul=str2.no_nul)
    //     pairtype(SomeInteger, SomeUnicodeString): -> SomeUnicodeString(no_nul=str2.no_nul)
    register(
        reg,
        OpKind::Mul,
        SomeValueTag::Integer,
        SomeValueTag::String,
        Specialization {
            apply: pure(integer_string_mul),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::Mul,
        SomeValueTag::Integer,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: pure(integer_unicodestring_mul),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn integer_string_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::String(s)) => s,
        _ => panic!("integer_string_mul: arg 1 not SomeString"),
    };
    SomeValue::String(SomeString::new(false, s2.inner.no_nul))
}

fn integer_unicodestring_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::UnicodeString(s)) => s,
        _ => panic!("integer_unicodestring_mul: arg 1 not SomeUnicodeString"),
    };
    SomeValue::UnicodeString(SomeUnicodeString::new(false, s2.inner.no_nul))
}

// binaryop.py __extend__ — pairtype(SomeUnicodeCodePoint, SomeUnicodeString)
//                     / (SomeUnicodeString, SomeUnicodeCodePoint)
//                     / (SomeUnicodeString, SomeUnicodeString): add

fn init_unicode_family_union_add(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // upstream:
    //     def add((str1, str2)):
    //         result = SomeUnicodeString(no_nul=str1.no_nul and str2.no_nul)
    //         if str1.is_immutable_constant() and str2.is_immutable_constant():
    //             result.const = str1.const + str2.const
    //         return result
    let pairs: &[(SomeValueTag, SomeValueTag)] = &[
        (SomeValueTag::UnicodeCodePoint, SomeValueTag::UnicodeString),
        (SomeValueTag::UnicodeString, SomeValueTag::UnicodeCodePoint),
        (SomeValueTag::UnicodeString, SomeValueTag::UnicodeString),
    ];
    for (t1, t2) in pairs {
        register(
            reg,
            OpKind::Add,
            *t1,
            *t2,
            Specialization {
                apply: pure(unicode_family_add),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

fn unicode_family_add(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    // Extract (no_nul, const) from either UnicodeString or UnicodeCodePoint.
    fn extract(sv: &SomeValue) -> (bool, Option<String>) {
        match sv {
            SomeValue::UnicodeString(s) => {
                let c = s
                    .inner
                    .base
                    .const_box
                    .as_ref()
                    .and_then(|c| match &c.value {
                        ConstValue::UniStr(v) => Some(v.clone()),
                        _ => None,
                    });
                (s.inner.no_nul, c)
            }
            SomeValue::UnicodeCodePoint(s) => {
                let c = s
                    .inner
                    .base
                    .const_box
                    .as_ref()
                    .and_then(|c| match &c.value {
                        ConstValue::UniStr(v) => Some(v.clone()),
                        _ => None,
                    });
                (s.inner.no_nul, c)
            }
            other => panic!("unicode_family_add: unexpected {:?}", other),
        }
    }
    let a = ann
        .annotation(&hl.args[0])
        .expect("unicode add: lhs unbound");
    let b = ann
        .annotation(&hl.args[1])
        .expect("unicode add: rhs unbound");
    let (n1, c1) = extract(&a);
    let (n2, c2) = extract(&b);
    let mut result = SomeUnicodeString::new(false, n1 && n2);
    if let (Some(a), Some(b)) = (c1, c2) {
        let combined = format!("{a}{b}");
        result.inner.base.const_box = Some(Constant::new(ConstValue::uni_str(combined)));
    }
    SomeValue::UnicodeString(result)
}

// binaryop.py:643-654 — @cmp_op.register(...) cross str/unicode comparisons

fn init_cmp_str_unicode(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // upstream:
    //     for cmp_op in [op.lt, op.le, op.eq, op.ne, op.gt, op.ge]:
    //         @cmp_op.register(SomeUnicodeString, SomeString)
    //         @cmp_op.register(SomeUnicodeString, SomeChar)
    //         @cmp_op.register(SomeString, SomeUnicodeString)
    //         @cmp_op.register(SomeChar, SomeUnicodeString)
    //         @cmp_op.register(SomeUnicodeCodePoint, SomeString)
    //         @cmp_op.register(SomeUnicodeCodePoint, SomeChar)
    //         @cmp_op.register(SomeString, SomeUnicodeCodePoint)
    //         @cmp_op.register(SomeChar, SomeUnicodeCodePoint)
    //         def cmp_str_unicode(...):
    //             raise AnnotatorError(
    //                 "Comparing byte strings with unicode strings is not RPython")
    let pairs: &[(SomeValueTag, SomeValueTag)] = &[
        (SomeValueTag::UnicodeString, SomeValueTag::String),
        (SomeValueTag::UnicodeString, SomeValueTag::Char),
        (SomeValueTag::String, SomeValueTag::UnicodeString),
        (SomeValueTag::Char, SomeValueTag::UnicodeString),
        (SomeValueTag::UnicodeCodePoint, SomeValueTag::String),
        (SomeValueTag::UnicodeCodePoint, SomeValueTag::Char),
        (SomeValueTag::String, SomeValueTag::UnicodeCodePoint),
        (SomeValueTag::Char, SomeValueTag::UnicodeCodePoint),
    ];
    for op in &[
        OpKind::Lt,
        OpKind::Le,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Gt,
        OpKind::Ge,
    ] {
        for (t1, t2) in pairs {
            register(
                reg,
                *op,
                *t1,
                *t2,
                Specialization {
                    apply: pure(cmp_str_unicode),
                    can_only_throw: CanOnlyThrow::Absent,
                },
            );
        }
    }
}

#[allow(non_snake_case)]
pub fn cmp_str_unicode(_ann: &RPythonAnnotator, _hl: &HLOperation) -> SomeValue {
    panic!("AnnotatorError: Comparing byte strings with unicode strings is not RPython")
}

// binaryop.py __extend__ — pairtype(SomeInteger, SomeList)

fn init_integer_list_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // upstream:
    //     def mul((int1, lst2)):
    //         bk = getbookkeeper()
    //         return lst2.listdef.offspring(bk)
    register(
        reg,
        OpKind::Mul,
        SomeValueTag::Integer,
        SomeValueTag::List,
        Specialization {
            apply: pure(integer_list_mul),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn integer_list_mul(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let lst2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("integer_list_mul: arg 1 not SomeList"),
    };
    let s_new = lst2
        .listdef
        .offspring(&ann.bookkeeper, &[])
        .expect("offspring failed");
    SomeValue::List(s_new)
}

// binaryop.py __extend__ — pairtype(SomeDict, SomeDict)

fn init_dict_dict_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `ne((dic1, dic2)):` raises AnnotatorError.
    register(
        reg,
        OpKind::Ne,
        SomeValueTag::Dict,
        SomeValueTag::Dict,
        Specialization {
            apply: pure(|_ann, _hl| panic!("AnnotatorError: dict != dict not implemented")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // `union` handled in model::union().
}

// binaryop.py — helpers + @op.getitem.register(SomeDict, SomeObject)
//
//     def _dict_can_only_throw_keyerror(s_dct, *ignore):
//         if s_dct.dictdef.dictkey.custom_eq_hash:
//             return None    # r_dict: can throw anything
//         return [KeyError]
//
//     def _dict_can_only_throw_nothing(s_dct, *ignore):
//         if s_dct.dictdef.dictkey.custom_eq_hash:
//             return None
//         return []
//
//     @op.getitem.register(SomeDict, SomeObject)
//     def getitem_SomeDict(annotator, v_dict, v_key):
//         s_dict = annotator.annotation(v_dict)
//         s_key = annotator.annotation(v_key)
//         s_dict.dictdef.generalize_key(s_key)
//         position = annotator.bookkeeper.position_key
//         return s_dict.dictdef.read_value(position)
//     getitem_SomeDict.can_only_throw = _dict_can_only_throw_keyerror

fn dict_can_only_throw_keyerror(args_s: &[SomeValue]) -> Option<Vec<BuiltinException>> {
    match args_s.first() {
        Some(SomeValue::Dict(d)) => {
            if d.dictdef.custom_eq_hash() {
                None
            } else {
                Some(vec![BuiltinException::KeyError])
            }
        }
        _ => Some(vec![BuiltinException::KeyError]),
    }
}

fn dict_can_only_throw_nothing(args_s: &[SomeValue]) -> Option<Vec<BuiltinException>> {
    match args_s.first() {
        Some(SomeValue::Dict(d)) => {
            if d.dictdef.custom_eq_hash() {
                None
            } else {
                Some(vec![])
            }
        }
        _ => Some(vec![]),
    }
}

fn init_dict_getitem(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::Dict,
        SomeValueTag::Object,
        Specialization {
            apply: pure(getitem_SomeDict),
            can_only_throw: CanOnlyThrow::Callable(Box::new(dict_can_only_throw_keyerror)),
        },
    );
}

#[allow(non_snake_case)]
pub fn getitem_SomeDict(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_dict = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::Dict(d)) => d,
        _ => panic!("getitem_SomeDict: arg 0 not SomeDict"),
    };
    let s_key = ann
        .annotation(&hl.args[1])
        .expect("dict getitem: key unbound");
    s_dict
        .dictdef
        .generalize_key(&s_key)
        .expect("getitem_SomeDict: generalize_key failed");
    // upstream: `position = annotator.bookkeeper.position_key`.
    // Outside a reflow frame this is `None`; `read_value` handles it.
    let position = ann.bookkeeper.current_position_key();
    s_dict.dictdef.read_value(position)
}

// binaryop.py __extend__ — pairtype(SomeDict, SomeObject)

fn init_dict_object_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `setitem((dic1, obj2), s_value)`:
    //     dic1.dictdef.generalize_key(obj2)
    //     dic1.dictdef.generalize_value(s_value)
    //     setitem.can_only_throw = _dict_can_only_throw_nothing
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::Dict,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(dict_object_setitem),
            can_only_throw: CanOnlyThrow::Callable(Box::new(dict_can_only_throw_nothing)),
        },
    );
    // binaryop.py — `delitem((dic1, obj2))`:
    //     dic1.dictdef.generalize_key(obj2)
    //     delitem.can_only_throw = _dict_can_only_throw_keyerror
    register(
        reg,
        OpKind::DelItem,
        SomeValueTag::Dict,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(dict_object_delitem),
            can_only_throw: CanOnlyThrow::Callable(Box::new(dict_can_only_throw_keyerror)),
        },
    );
}

fn dict_object_setitem(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    let dic1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::Dict(d)) => d,
        _ => panic!("dict_object_setitem: arg 0 not SomeDict"),
    };
    let s_key = ann
        .annotation(&hl.args[1])
        .expect("dict setitem: key unbound");
    let s_value = ann
        .annotation(&hl.args[2])
        .expect("dict setitem: value unbound");
    dic1.dictdef
        .generalize_key(&s_key)
        .expect("generalize_key failed");
    dic1.dictdef
        .generalize_value(&s_value)
        .expect("generalize_value failed");
    // upstream `setitem` returns None (void).
    None
}

fn dict_object_delitem(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    let dic1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::Dict(d)) => d,
        _ => panic!("dict_object_delitem: arg 0 not SomeDict"),
    };
    let s_key = ann
        .annotation(&hl.args[1])
        .expect("dict delitem: key unbound");
    dic1.dictdef
        .generalize_key(&s_key)
        .expect("generalize_key failed");
    // upstream `delitem` returns None (void).
    None
}

// binaryop.py __extend__ — pairtype(SomeIterator, SomeIterator)

fn init_iterator_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream only defines `union` — handled in model::union().
}

// binaryop.py __extend__ — pairtype(SomeBuiltinMethod, SomeBuiltinMethod)

fn init_builtinmethod_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Upstream only defines `union` — handled in model::union().
}

// binaryop.py:768-780 — @op.is_.register(SomePBC, SomePBC)
//
//     def is__PBC_PBC(annotator, pbc1, pbc2):
//         s = is__default(annotator, pbc1, pbc2)
//         if not s.is_constant():
//             s_pbc1 = annotator.annotation(pbc1)
//             s_pbc2 = annotator.annotation(pbc2)
//             if not s_pbc1.can_be_None or not s_pbc2.can_be_None:
//                 for desc in s_pbc1.descriptions:
//                     if desc in s_pbc2.descriptions:
//                         break
//                 else:
//                     s.const = False    # no common desc in the two sets
//         return s

fn init_pbc_pbc_is_(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    register(
        reg,
        OpKind::Is,
        SomeValueTag::PBC,
        SomeValueTag::PBC,
        Specialization {
            apply: pure(is__PBC_PBC),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

#[allow(non_snake_case)]
pub fn is__PBC_PBC(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    // Upstream delegates to `is__default` then refines the `const`
    // attribute when the two PBC sets are disjoint. The default is a
    // SomeBool, not necessarily constant.
    //
    // `is__default` in Rust port lives in init_is_default — call it
    // by going through consider once; here we inline the minimal logic
    // (non-constant SomeBool) and then apply the refinement step.
    let s_pbc1 = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::PBC(p)) => p,
        _ => panic!("is__PBC_PBC: arg 0 not SomePBC"),
    };
    let s_pbc2 = match ann.annotation(&hl.args[1]) {
        Some(SomeValue::PBC(p)) => p,
        _ => panic!("is__PBC_PBC: arg 1 not SomePBC"),
    };
    let mut s = SomeBool::new();
    if !s_pbc1.can_be_none || !s_pbc2.can_be_none {
        // upstream: `pbc1.descriptions & pbc2.descriptions` — set
        //   intersection by identity. Rust port keys on DescKey.
        let common = s_pbc1
            .descriptions
            .keys()
            .any(|k| s_pbc2.descriptions.contains_key(k));
        if !common {
            // upstream: `s.const = False`.
            s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        }
    }
    SomeValue::Bool(s)
}

// binaryop.py — Impossible/None union is handled via model::union().
// This init is a no-op guard placeholder — those pair types only bind
// `union`, which the union dispatch already covers.

fn init_impossible_none_pairtype(
    _reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // Handled in model::union() — lines 788-812 upstream only defined
    // `union` methods:
    //   pairtype(SomeImpossibleValue, SomeObject) / reverse
    //   pairtype(SomeObject, SomeNone) / (SomeNone, SomeObject)
    //   pairtype(SomeImpossibleValue, SomeNone) / reverse
}

// binaryop.py __extend__ — pairtype(SomePBC, SomeObject)

fn init_pbc_object_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `getitem` / `setitem` raise AnnotatorError.
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::PBC,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, hl| panic!("AnnotatorError: getitem on {:?}", hl.args[0])),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::PBC,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, hl| panic!("AnnotatorError: setitem on {:?}", hl.args[0])),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// binaryop.py __extend__ — pairtype(SomeNone, SomeObject)

fn init_none_object_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — `getitem((none, o))`:
    //     return s_ImpossibleValue
    //     getitem.can_only_throw = []
    register(
        reg,
        OpKind::GetItem,
        SomeValueTag::None_,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // binaryop.py — `setitem((none, o), s_value)`: return None (void).
    register(
        reg,
        OpKind::SetItem,
        SomeValueTag::None_,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| None),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// binaryop.py __extend__ — pairtype(SomePBC|SomeNone, SomeString) / reverse

fn init_pbc_string_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — pairtype(SomePBC, SomeString).add: AnnotatorError
    register(
        reg,
        OpKind::Add,
        SomeValueTag::PBC,
        SomeValueTag::String,
        Specialization {
            apply: pure(|_ann, hl| panic!("AnnotatorError: add on {:?}", hl.args[0])),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // binaryop.py — pairtype(SomeString, SomePBC).add: AnnotatorError
    register(
        reg,
        OpKind::Add,
        SomeValueTag::String,
        SomeValueTag::PBC,
        Specialization {
            apply: pure(|_ann, hl| panic!("AnnotatorError: add on {:?}", hl.args[1])),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// binaryop.py / 685-708 — `pair(...).improve()` free function
//
// `improve` is NOT an OpKind — `pair(a, b).improve()` is invoked
// directly by the annotator's flowin/isinstance-branch refinement
// (annrpython.py), not dispatched through the operation registry.
// These helpers stay available for that call site once Commit 7/8
// wires up flowin. Upstream lives on SomeObject (default) and
// SomeInstance (pairtype override) in binaryop.py.

/// RPython `pairtype(SomeObject, SomeObject).improve((obj, improvement))`
/// (binaryop.py):
///
/// ```python
/// if not improvement.contains(obj) and obj.contains(improvement):
///     return improvement
/// else:
///     return obj
/// ```
fn improve_default(obj: &SomeValue, improvement: &SomeValue) -> SomeValue {
    if !improvement.contains(obj) && obj.contains(improvement) {
        improvement.clone()
    } else {
        obj.clone()
    }
}

/// RPython `pairtype(SomeInstance, SomeInstance).improve((ins1, ins2))`
/// (binaryop.py). Falls back to [`improve_default`] when either
/// side is not `SomeInstance`.
pub(crate) fn improve(obj: &SomeValue, improvement: &SomeValue) -> SomeValue {
    if let (SomeValue::Instance(ins1), SomeValue::Instance(ins2)) = (obj, improvement) {
        return improve_instance(ins1, ins2);
    }
    improve_default(obj, improvement)
}

fn improve_instance(ins1: &model::SomeInstance, ins2: &model::SomeInstance) -> SomeValue {
    use super::classdesc::ClassDef;
    use model::SomeInstance;
    use std::cell::RefCell;
    // upstream binaryop.py: classdef refinement + super().improve() fallback.
    let resdef: Option<Rc<RefCell<ClassDef>>> = match (&ins1.classdef, &ins2.classdef) {
        (None, _) => ins2.classdef.clone(),
        (_, None) => ins1.classdef.clone(),
        (Some(ca), Some(cb)) => {
            let basedef = ClassDef::commonbase(ca, cb);
            match basedef {
                Some(bd) if Rc::ptr_eq(&bd, ca) => Some(cb.clone()),
                Some(bd) if Rc::ptr_eq(&bd, cb) => Some(ca.clone()),
                _ => {
                    if ins1.can_be_none && ins2.can_be_none {
                        return model::s_none();
                    }
                    return model::s_impossible_value();
                }
            }
        }
    };
    let res = SomeInstance::new(
        resdef,
        ins1.can_be_none && ins2.can_be_none,
        std::collections::BTreeMap::new(),
    );
    let res_sv = SomeValue::Instance(res);
    let a_sv = SomeValue::Instance(ins1.clone());
    let b_sv = SomeValue::Instance(ins2.clone());
    if a_sv.contains(&res_sv) && b_sv.contains(&res_sv) {
        res_sv
    } else {
        improve_default(&a_sv, &b_sv)
    }
}

fn init_none_string_pairtype(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Specialization>>,
) {
    // binaryop.py — pairtype(SomeNone, SomeString).add: s_ImpossibleValue
    register(
        reg,
        OpKind::Add,
        SomeValueTag::None_,
        SomeValueTag::String,
        Specialization {
            apply: pure(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // binaryop.py — pairtype(SomeString, SomeNone).add: s_ImpossibleValue
    register(
        reg,
        OpKind::Add,
        SomeValueTag::String,
        SomeValueTag::None_,
        Specialization {
            apply: pure(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// binaryop.py:727-747 — pairtype(SomeInstance, SomeObject) transforms

/// Build a new [`HLOperation`] with a fresh result [`Variable`] — the
/// Rust equivalent of upstream `op.<name>(args...)` inside a transform
/// handler. Upstream relies on `HLOperation.__init__` allocating a
/// fresh `Variable()` (operation.py:73-75).
fn mk_hlop(kind: OpKind, args: Vec<Hlvalue>) -> HLOperation {
    HLOperation::new(kind, args)
}

/// RPython `@op.getitem.register_transform(SomeInstance, SomeObject)`
/// / setitem / delitem / contains (binaryop.py). All four
/// rewrite `op(v_ins, ...)` to `[op.getattr(v_ins, '__name__'),
/// op.simple_call(getattr.result, ...)]`.
fn init_instance_object_transform(
    reg: &mut HashMap<OpKind, DoubleDispatchRegistry<SomeValueTag, SomeValueTag, Transformation>>,
) {
    // binaryop.py — getitem
    register_transform(
        reg,
        OpKind::GetItem,
        SomeValueTag::Instance,
        SomeValueTag::Object,
        Box::new(getitem_SomeInstance),
    );
    // binaryop.py — setitem
    register_transform(
        reg,
        OpKind::SetItem,
        SomeValueTag::Instance,
        SomeValueTag::Object,
        Box::new(setitem_SomeInstance),
    );
    // binaryop.py — delitem
    register_transform(
        reg,
        OpKind::DelItem,
        SomeValueTag::Instance,
        SomeValueTag::Object,
        Box::new(delitem_SomeInstance),
    );
    // binaryop.py:744-747 — `contains` is Dispatch::Single upstream;
    // handler lives in unaryop::init_transform below.
}

pub(crate) fn init_contains_instance_transform(
    reg: &mut HashMap<OpKind, HashMap<SomeValueTag, Transformation>>,
) {
    reg.entry(OpKind::Contains)
        .or_default()
        .insert(SomeValueTag::Instance, Box::new(contains_SomeInstance));
}

#[allow(non_snake_case)]
pub fn getitem_SomeInstance(_ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_ins = args[0].clone();
    let v_idx = args[1].clone();
    let get_getitem = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_ins,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__getitem__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_getitem.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_idx]);
    Some(vec![get_getitem, call])
}

#[allow(non_snake_case)]
pub fn setitem_SomeInstance(_ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_ins = args[0].clone();
    let v_idx = args[1].clone();
    let v_value = args[2].clone();
    let get_setitem = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_ins,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__setitem__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_setitem.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_idx, v_value]);
    Some(vec![get_setitem, call])
}

#[allow(non_snake_case)]
pub fn delitem_SomeInstance(_ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_ins = args[0].clone();
    let v_idx = args[1].clone();
    let get_delitem = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_ins,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__delitem__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_delitem.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_idx]);
    Some(vec![get_delitem, call])
}

#[allow(non_snake_case)]
pub fn contains_SomeInstance(
    _ann: &RPythonAnnotator,
    args: &[Hlvalue],
) -> Option<Vec<HLOperation>> {
    let v_ins = args[0].clone();
    let v_idx = args[1].clone();
    let get_contains = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_ins,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__contains__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_contains.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_idx]);
    Some(vec![get_contains, call])
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::argument::Signature;
    use super::super::super::flowspace::operation::{Dispatch, OpKind};
    use super::*;
    use crate::annotator::description::{DescEntry, FunctionDesc};
    use std::cell::RefCell;

    #[test]
    fn binary_operations_includes_add_sub_mul() {
        assert!(BINARY_OPERATIONS.contains(&OpKind::Add));
        assert!(BINARY_OPERATIONS.contains(&OpKind::Sub));
        assert!(BINARY_OPERATIONS.contains(&OpKind::Mul));
        assert!(BINARY_OPERATIONS.contains(&OpKind::Is));
        assert!(BINARY_OPERATIONS.contains(&OpKind::GetItem));
    }

    #[test]
    fn binary_operations_excludes_unary() {
        assert!(!BINARY_OPERATIONS.contains(&OpKind::Neg));
        assert!(!BINARY_OPERATIONS.contains(&OpKind::Len));
    }

    #[test]
    fn host_to_annotation_spec_maps_primitives() {
        use super::super::super::flowspace::model::HostObject;
        use super::super::signature::AnnotationSpec;
        let primitives = [
            ("bool", AnnotationSpec::Bool),
            ("int", AnnotationSpec::Int),
            ("float", AnnotationSpec::Float),
            ("str", AnnotationSpec::Str),
            ("unicode", AnnotationSpec::Unicode),
            ("NoneType", AnnotationSpec::NoneType),
            ("type", AnnotationSpec::Type),
        ];
        for (name, expected) in primitives {
            let cv = ConstValue::HostObject(HostObject::new_class(name, vec![]));
            assert_eq!(
                host_to_annotation_spec(&cv),
                Some(expected),
                "primitive {name} must map to AnnotationSpec"
            );
        }
    }

    #[test]
    fn host_to_annotation_spec_maps_user_class() {
        use super::super::super::flowspace::model::HostObject;
        use super::super::signature::AnnotationSpec;
        let cls = HostObject::new_class("pkg.MyClass", vec![]);
        let cv = ConstValue::HostObject(cls.clone());
        let got = host_to_annotation_spec(&cv).expect("user class must map");
        match got {
            AnnotationSpec::UserClass(h) => assert_eq!(h.qualname(), "pkg.MyClass"),
            other => panic!("expected UserClass, got {other:?}"),
        }
    }

    #[test]
    fn host_to_annotation_spec_rejects_non_host_constants() {
        assert!(host_to_annotation_spec(&ConstValue::Int(42)).is_none());
        assert!(host_to_annotation_spec(&ConstValue::Bool(true)).is_none());
        assert!(host_to_annotation_spec(&ConstValue::None).is_none());
    }

    #[test]
    fn binary_operations_matches_dispatch_double() {
        for op in BINARY_OPERATIONS {
            assert!(
                matches!(op.dispatch(), Dispatch::Double),
                "{:?} in BINARY_OPERATIONS but dispatch() is not Double",
                op
            );
        }
    }

    fn mk_ann() -> Rc<RPythonAnnotator> {
        RPythonAnnotator::new(None, None, None, false)
    }

    fn fake_function_entry(bk: &Rc<super::super::bookkeeper::Bookkeeper>, name: &str) -> DescEntry {
        DescEntry::function(Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            name,
            Signature::new(vec![], None, None),
            None,
            None,
        ))))
    }

    fn hl_int_int(op: OpKind) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = mk_ann();
        let v0 = Variable::named("a");
        let v1 = Variable::named("b");
        let mut v0 = v0;
        let mut v1 = v1;
        ann.setbinding(&mut v0, SomeValue::Integer(SomeInteger::default()));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(op, vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)]);
        (hl, ann)
    }

    #[test]
    fn consider_integer_add_returns_someinteger() {
        let (hl, ann) = hl_int_int(OpKind::Add);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_integer_truediv_returns_somefloat() {
        let (hl, ann) = hl_int_int(OpKind::TrueDiv);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn get_can_only_throw_integer_div_is_zerodivision() {
        let (hl, ann) = hl_int_int(OpKind::Div);
        let cot = hl.get_can_only_throw(&ann);
        assert_eq!(cot, Some(vec![BuiltinException::ZeroDivisionError]));
    }

    #[test]
    fn consider_integer_lt_returns_somebool() {
        let (base, ann) = hl_int_int(OpKind::Add);
        let hl = HLOperation::new(OpKind::Lt, base.args.clone());
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    fn hl_bool_bool(op: OpKind) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let v0 = Variable::named("a");
        let v1 = Variable::named("b");
        let mut v0 = v0;
        let mut v1 = v1;
        ann.setbinding(&mut v0, SomeValue::Bool(SomeBool::new()));
        ann.setbinding(&mut v1, SomeValue::Bool(SomeBool::new()));
        let hl = HLOperation::new(op, vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)]);
        (hl, ann)
    }

    #[test]
    fn consider_bool_and_returns_somebool() {
        let (hl, ann) = hl_bool_bool(OpKind::And);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_bool_xor_returns_somebool() {
        let (hl, ann) = hl_bool_bool(OpKind::Xor);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    fn hl_float_float(op: OpKind) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let v0 = Variable::named("a");
        let v1 = Variable::named("b");
        let mut v0 = v0;
        let mut v1 = v1;
        ann.setbinding(&mut v0, SomeValue::Float(SomeFloat::new()));
        ann.setbinding(&mut v1, SomeValue::Float(SomeFloat::new()));
        let hl = HLOperation::new(op, vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)]);
        (hl, ann)
    }

    #[test]
    fn consider_float_div_returns_somefloat() {
        let (hl, ann) = hl_float_float(OpKind::Div);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_float_add_returns_somefloat() {
        let (hl, ann) = hl_float_float(OpKind::Add);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_inplace_add_reifies_nested_binary_dispatch_errors() {
        let ann = mk_ann();
        let mut v0 = Variable::named("pbc");
        let mut v1 = Variable::named("s");
        ann.setbinding(
            &mut v0,
            SomeValue::PBC(super::super::model::SomePBC::new(
                vec![fake_function_entry(&ann.bookkeeper, "f")],
                false,
            )),
        );
        ann.setbinding(&mut v1, SomeValue::String(SomeString::new(false, false)));
        let hl = HLOperation::new(
            OpKind::InplaceAdd,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );

        let err = hl
            .consider(&ann)
            .expect_err("nested binary dispatch AnnotatorError must be reified");
        assert!(err.msg.as_deref().unwrap_or("").contains("add on"));
    }

    fn hl_string_string_const(c1: &str, c2: &str) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = mk_ann();
        let mut v0 = Variable::named("s0");
        let mut v1 = Variable::named("s1");
        let mut s0 = SomeString::new(false, true);
        s0.inner.base.const_box = Some(Constant::new(ConstValue::byte_str(c1)));
        let mut s1 = SomeString::new(false, true);
        s1.inner.base.const_box = Some(Constant::new(ConstValue::byte_str(c2)));
        ann.setbinding(&mut v0, SomeValue::String(s0));
        ann.setbinding(&mut v1, SomeValue::String(s1));
        let hl = HLOperation::new(
            OpKind::Add,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        (hl, ann)
    }

    #[test]
    fn consider_string_add_propagates_const() {
        // binaryop.py add — const+const should yield const=str1+str2.
        let (hl, ann) = hl_string_string_const("foo", "bar");
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::String(s) => {
                let c = s.inner.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::byte_str("foobar"));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_string_char_add_uses_somechar_string_mro() {
        for char_first in [false, true] {
            let ann = mk_ann();
            let mut string_var = Variable::named("s");
            let mut char_var = Variable::named("c");
            let mut string = SomeString::new(false, true);
            string.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("left")));
            let mut character = SomeChar::new(true);
            character.inner.base.const_box = Some(Constant::new(ConstValue::byte_str(" ")));
            ann.setbinding(&mut string_var, SomeValue::String(string));
            ann.setbinding(&mut char_var, SomeValue::Char(character));
            let args = if char_first {
                vec![Hlvalue::Variable(char_var), Hlvalue::Variable(string_var)]
            } else {
                vec![Hlvalue::Variable(string_var), Hlvalue::Variable(char_var)]
            };
            let result = HLOperation::new(OpKind::Add, args)
                .consider(&ann)
                .unwrap()
                .unwrap();
            let SomeValue::String(result) = result else {
                panic!("string/char add must widen to SomeString")
            };
            assert_eq!(
                result.inner.base.const_box.expect("constant add").value,
                ConstValue::byte_str(if char_first { " left" } else { "left " })
            );
        }
    }

    #[test]
    fn consider_bytearray_add_returns_bytearray() {
        let ann = mk_ann();
        let mut v0 = Variable::named("b0");
        let mut v1 = Variable::named("b1");
        ann.setbinding(&mut v0, SomeValue::ByteArray(SomeByteArray::new(false)));
        ann.setbinding(&mut v1, SomeValue::ByteArray(SomeByteArray::new(false)));
        let hl = HLOperation::new(
            OpKind::Add,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::ByteArray(_)), "got {:?}", r);
    }

    #[test]
    fn consider_string_bytearray_add_returns_bytearray() {
        // binaryop.py — cross pair add → SomeByteArray.
        let ann = mk_ann();
        let mut v0 = Variable::named("s");
        let mut v1 = Variable::named("b");
        ann.setbinding(&mut v0, SomeValue::String(SomeString::new(false, false)));
        ann.setbinding(&mut v1, SomeValue::ByteArray(SomeByteArray::new(false)));
        let hl = HLOperation::new(
            OpKind::Add,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::ByteArray(_)), "got {:?}", r);
    }

    #[test]
    fn consider_bytearray_int_getitem_returns_someinteger() {
        let ann = mk_ann();
        let mut v0 = Variable::named("b");
        let mut v1 = Variable::named("i");
        ann.setbinding(&mut v0, SomeValue::ByteArray(SomeByteArray::new(false)));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_tuple_tuple_add_concats_items() {
        // binaryop.py — add returns SomeTuple with concatenated items.
        let ann = mk_ann();
        let mut v0 = Variable::named("t0");
        let mut v1 = Variable::named("t1");
        ann.setbinding(
            &mut v0,
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
        );
        ann.setbinding(
            &mut v1,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Bool(SomeBool::new()),
                SomeValue::Float(SomeFloat::new()),
            ])),
        );
        let hl = HLOperation::new(
            OpKind::Add,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Tuple(t) => assert_eq!(t.items.len(), 3),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_tuple_integer_getitem_constant_picks_item() {
        // binaryop.py getitem — constant index path returns items[const].
        let ann = mk_ann();
        let mut v0 = Variable::named("t");
        let mut v1 = Variable::named("i");
        ann.setbinding(
            &mut v0,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::String(SomeString::new(false, false)),
            ])),
        );
        let mut const_idx = SomeInteger::default();
        const_idx.base.const_box = Some(Constant::new(ConstValue::Int(1)));
        ann.setbinding(&mut v1, SomeValue::Integer(const_idx));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_tuple_integer_getitem_nonconst_unions() {
        // binaryop.py:567-568 — nonconst index returns unionof(*items).
        let ann = mk_ann();
        let mut v0 = Variable::named("t");
        let mut v1 = Variable::named("i");
        ann.setbinding(
            &mut v0,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_string_integer_getitem_returns_somechar() {
        // binaryop.py getitem — returns SomeChar(no_nul=str1.no_nul).
        let ann = mk_ann();
        let mut v0 = Variable::named("s");
        let mut v1 = Variable::named("i");
        ann.setbinding(&mut v0, SomeValue::String(SomeString::new(false, true)));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Char(c) => assert!(c.inner.no_nul, "no_nul propagated"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_unicodestring_integer_getitem_returns_unicodecp() {
        let ann = mk_ann();
        let mut v0 = Variable::named("s");
        let mut v1 = Variable::named("i");
        ann.setbinding(
            &mut v0,
            SomeValue::UnicodeString(SomeUnicodeString::new(false, true)),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::UnicodeCodePoint(_)), "got {:?}", r);
    }

    #[test]
    fn consider_integer_string_mul_returns_string() {
        let ann = mk_ann();
        let mut v0 = Variable::named("i");
        let mut v1 = Variable::named("s");
        ann.setbinding(&mut v0, SomeValue::Integer(SomeInteger::default()));
        ann.setbinding(&mut v1, SomeValue::String(SomeString::new(false, false)));
        let hl = HLOperation::new(
            OpKind::Mul,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_ptr_integer_getitem_returns_low_level_item_annotation() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let mut v0 = Variable::named("p");
        let mut v1 = Variable::named("i");
        ann.setbinding(
            &mut v0,
            SomeValue::Ptr(model::SomePtr::new(Ptr {
                TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
            })),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_ptr_integer_getitem_zero_length_fixed_array_returns_none() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FixedSizeArray, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let mut v0 = Variable::named("p");
        let mut v1 = Variable::named("i");
        ann.setbinding(
            &mut v0,
            SomeValue::Ptr(model::SomePtr::new(Ptr {
                TO: PtrTarget::FixedSizeArray(FixedSizeArray::new(LowLevelType::Signed, 0)),
            })),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        // llannotation.py — getitem on a zero-length array hits
        // IndexError and returns None (void); consider binds the result to
        // Impossible without blocking, so the raw consider yields None.
        let r = hl.consider(&ann).unwrap();
        assert!(
            r.is_none(),
            "void out-of-bounds getitem should be None, got {:?}",
            r
        );
    }

    #[test]
    fn consider_ptr_integer_setitem_checks_low_level_item_type() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let mut v0 = Variable::named("p");
        let mut v1 = Variable::named("i");
        let mut v2 = Variable::named("v");
        ann.setbinding(
            &mut v0,
            SomeValue::Ptr(model::SomePtr::new(Ptr {
                TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
            })),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        ann.setbinding(&mut v2, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::SetItem,
            vec![
                Hlvalue::Variable(v0),
                Hlvalue::Variable(v1),
                Hlvalue::Variable(v2),
            ],
        );
        // llannotation.py — ptr setitem is a void op: consider
        // returns None (the result var is bound to Impossible without
        // blocking).
        let r = hl.consider(&ann).unwrap();
        assert!(
            r.is_none(),
            "void ptr setitem should return None, got {:?}",
            r
        );
    }

    #[test]
    fn consider_interiorptr_integer_getitem_returns_low_level_item_annotation() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, InteriorOffset, InteriorPtr, LowLevelType, Struct,
        };

        let ann = mk_ann();
        let mut v0 = Variable::named("p");
        let mut v1 = Variable::named("i");
        ann.setbinding(
            &mut v0,
            SomeValue::InteriorPtr(model::SomeInteriorPtr::new(InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(Struct::new(
                    "S",
                    vec![(
                        "arr".into(),
                        LowLevelType::Array(Box::new(Array::new(LowLevelType::Signed))),
                    )],
                )))),
                TO: Box::new(LowLevelType::Array(Box::new(Array::new(
                    LowLevelType::Signed,
                )))),
                offsets: vec![InteriorOffset::Field("arr".into())],
            })),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_ptr_object_getitem_rejects_non_integer_index() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let mut v0 = Variable::named("p");
        let mut v1 = Variable::named("o");
        ann.setbinding(
            &mut v0,
            SomeValue::Ptr(model::SomePtr::new(Ptr {
                TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
            })),
        );
        ann.setbinding(&mut v1, SomeValue::object());
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let err = hl
            .consider(&ann)
            .expect_err("ptr getitem with non-integer index must raise AnnotatorError");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("getitem index not an int")
        );
    }

    #[test]
    fn consider_list_integer_getitem_accepts_none_position() {
        // reviewer parity — upstream allows position_key=None as a dict
        // key (Python dict accepts None); read_item drops None from
        // read_locations but still returns the item annotation.
        let ann = mk_ann();
        // Do NOT set_position_key — deliberately leave it as None.
        let mut v0 = Variable::named("lst");
        let mut v1 = Variable::named("i");
        let listdef = super::super::listdef::ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        );
        let s_list = super::super::model::SomeList::new(listdef);
        ann.setbinding(&mut v0, SomeValue::List(s_list));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(
            matches!(r, SomeValue::Impossible | SomeValue::Integer(_)),
            "got {:?}",
            r
        );
    }

    #[test]
    fn consider_none_object_getitem_returns_impossible() {
        // binaryop.py — getitem((none, o)) returns s_ImpossibleValue.
        let ann = mk_ann();
        let mut v0 = Variable::named("n");
        let mut v1 = Variable::named("o");
        ann.setbinding(&mut v0, SomeValue::None_(super::model::SomeNone::new()));
        ann.setbinding(&mut v1, SomeValue::object());
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_none_string_add_returns_impossible() {
        // binaryop.py — pairtype(SomeNone, SomeString).add → s_ImpossibleValue.
        let ann = mk_ann();
        let mut v0 = Variable::named("n");
        let mut v1 = Variable::named("s");
        ann.setbinding(&mut v0, SomeValue::None_(super::model::SomeNone::new()));
        ann.setbinding(&mut v1, SomeValue::String(SomeString::new(false, false)));
        let hl = HLOperation::new(
            OpKind::Add,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn improve_default_equal_returns_obj() {
        // binaryop.py improve — when obj == improvement, contains() is
        // reflexive so the guard `not improvement.contains(obj)` is
        // False and we return obj.
        let obj = SomeValue::Integer(SomeInteger::default());
        let improvement = SomeValue::Integer(SomeInteger::default());
        let r = improve(&obj, &improvement);
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn improve_default_bool_to_integer_keeps_bool() {
        // improve(SomeBool, SomeInteger): SomeBool ⊂ SomeInteger, so
        // obj.contains(improvement) = False (SomeBool cannot contain
        // SomeInteger). The else branch returns obj.
        let obj = SomeValue::Bool(SomeBool::new());
        let improvement = SomeValue::Integer(SomeInteger::default());
        let r = improve(&obj, &improvement);
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn improve_instance_none_classdef_picks_other_side() {
        // binaryop.py:686-689 — if ins1.classdef is None, resdef = ins2.classdef.
        let ins1 = super::model::SomeInstance::new(None, false, std::collections::BTreeMap::new());
        let ins2 = super::model::SomeInstance::new(None, true, std::collections::BTreeMap::new());
        let r = improve(&SomeValue::Instance(ins1), &SomeValue::Instance(ins2));
        match r {
            SomeValue::Instance(x) => {
                assert!(x.classdef.is_none());
                // can_be_none is ins1.can_be_None AND ins2.can_be_None = false AND true = false.
                assert!(!x.can_be_none);
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_dict_object_getitem_returns_value_type() {
        // binaryop.py — getitem_SomeDict returns dictdef.read_value
        // at bookkeeper.position_key; we check the return path roots at
        // the current value annotation by installing a minimal position_key.
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(7, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        // Build a SomeDict with int key + str value.
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::model::SomeDict::new(dictdef);
        let mut v0 = Variable::named("d");
        let mut v1 = Variable::named("k");
        ann.setbinding(&mut v0, SomeValue::Dict(s_dict));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }
}
