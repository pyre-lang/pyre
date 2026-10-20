//! RPython `rpython/annotator/unaryop.py` — single-dispatch
//! (`dispatch == 1`) operation handlers on the `SomeValue` lattice.
//!
//! Commit split:
//!
//! * **Commit 6a (this file initial drop)** — `UNARY_OPERATIONS`
//!   listing + `@op.type.register(SomeObject)` +
//!   `@op.contains.register(SomeNone|SomeInteger|SomeFloat|SomeBool)` +
//!   simple-returning `class __extend__(SomeObject)` defaults +
//!   primitive overrides on `SomeFloat` / `SomeInteger` / `SomeBool` +
//!   `class __extend__(SomeTuple)` unary methods.
//! * **Commit 6a2** — `@op.bool.register(SomeObject)` with knowntypedata
//!   branch refinement, `@op.isinstance.register` /
//!   `simple_call` / `call_args` / `issubtype` / `bool_behavior` /
//!   `getattr` / `setattr` / `hash` refinement with `find_method`.
//! * **Commit 6b** — list/dict `method_*` analyzers.
//! * **Commit 6c** — string/bytearray/char/unicode `method_*` analyzers.
//! * **Commit 6d** — Instance / Iterator / Builtin / PBC / None /
//!   Exception / WeakRef handlers.

use std::rc::Rc;

use super::super::flowspace::model::ConstValue;
use super::super::flowspace::model::Constant;
use super::super::flowspace::model::HOST_ENV;
use super::super::flowspace::model::Hlvalue;
use super::super::flowspace::model::HostObject;
use super::super::flowspace::operation::{
    BuiltinException, CanOnlyThrow, HLOperation, OpKind, Specialization, Transformation, pure,
};
use super::annrpython::RPythonAnnotator;
use super::classdesc::{BuiltinTypeDesc, ClassDesc, is_primitive_type};
use super::model::{
    AnnotatorError, ExitCaseKey, KnownType, SomeBool, SomeBuiltinMethod, SomeFloat, SomeInteger,
    SomeIterator, SomeObjectTrait, SomeRustVec, SomeString, SomeTuple, SomeTypeOf,
    SomeUnicodeString, SomeValue, SomeValueTag, add_knowntypedata, s_bool, s_impossible_value,
    s_none,
};

/// RPython `UNARY_OPERATIONS` (unaryop.py).
///
/// ```python
/// UNARY_OPERATIONS = set([oper.opname for oper in op.__dict__.values()
///                         if oper.dispatch == 1])
/// UNARY_OPERATIONS.remove('contains')
/// ```
pub static UNARY_OPERATIONS: &[OpKind] = &[
    OpKind::Id,
    OpKind::Type,
    OpKind::IsSubtype,
    OpKind::IsInstance,
    OpKind::Repr,
    OpKind::Str,
    OpKind::Len,
    OpKind::Hash,
    OpKind::SetAttr,
    OpKind::DelAttr,
    OpKind::GetSlice,
    OpKind::SetSlice,
    OpKind::DelSlice,
    OpKind::Pos,
    OpKind::Neg,
    OpKind::NegOvf,
    OpKind::Bool,
    OpKind::Abs,
    OpKind::AbsOvf,
    OpKind::Hex,
    OpKind::Oct,
    OpKind::Bin,
    OpKind::Ord,
    OpKind::Invert,
    OpKind::Int,
    OpKind::Float,
    OpKind::Long,
    OpKind::Hint,
    OpKind::Iter,
    OpKind::Next,
    OpKind::GetAttr,
    OpKind::SimpleCall,
    OpKind::CallArgs,
    // `Contains` is removed from UNARY_OPERATIONS upstream line 28 —
    // it's a binary op on the `in` operand semantically. The `@op.contains.register`
    // hooks below still live in unaryop.py, so they land here.
];

/// Module-import-time population of `_REGISTRY_SINGLE` — mirrors the
/// side-effect of RPython importing unaryop.py (which runs all the
/// `@op.X.register(Some_cls)` decorators at import time).
pub(crate) fn init(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    init_type_register(reg);
    init_contains_register(reg);
    init_someobject_defaults(reg);
    init_somefloat_overrides(reg);
    init_someinteger_overrides(reg);
    init_somebool_overrides(reg);
    init_sometuple_overrides(reg);
    init_somelist_overrides(reg);
    init_somerustvec_overrides(reg);
    init_somedict_overrides(reg);
    init_somestring_overrides(reg);
    init_someunicodestring_overrides(reg);
    init_somebytearray_overrides(reg);
    init_somechar_overrides(reg);
    init_someunicodecp_overrides(reg);
    init_someiterator_overrides(reg);
    init_somepbc_overrides(reg);
    init_someptr_overrides(reg);
    init_somenone_overrides(reg);
    init_someweakref_overrides(reg);
    init_someinstance_overrides(reg);
}

/// Module-import time side effect mirroring
/// `@op.<name>.register_transform(...)` decorators in unaryop.py. Runs
/// at `_TRANSFORM_SINGLE` thread_local init; one block per upstream
/// `@op.X.register_transform(Y)` decorator.
pub(crate) fn init_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    init_instance_single_transform(reg);
    init_instance_attr_transform(reg);
    init_object_call_args_transform(reg);
}

fn register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
    op: OpKind,
    tag: SomeValueTag,
    spec: Specialization,
) {
    reg.entry(op).or_default().insert(tag, spec);
}

fn register_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
    op: OpKind,
    tag: SomeValueTag,
    tx: Transformation,
) {
    reg.entry(op).or_default().insert(tag, tx);
}

// unaryop.py:31-33 — @op.type.register(SomeObject)
//
//     @op.type.register(SomeObject)
//     def type_SomeObject(annotator, v_arg):
//         return SomeTypeOf([v_arg])

#[allow(non_snake_case)]
pub fn type_SomeObject(_ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    // upstream unaryop.py — `SomeTypeOf([v_arg])`.
    // Constants are ignored: upstream `type(const)` would never enter
    // this dispatcher (it lands on the const-folding path of
    // `Type.constfold`). When the arg happens to be a constant we fall
    // back to an empty is_type_of list.
    let vars: Vec<Rc<super::super::flowspace::model::Variable>> = match &hl.args[0] {
        super::super::flowspace::model::Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
        super::super::flowspace::model::Hlvalue::Constant(_) => Vec::new(),
    };
    SomeValue::TypeOf(SomeTypeOf::new(vars))
}

fn init_type_register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    register(
        reg,
        OpKind::Type,
        SomeValueTag::Object,
        Specialization {
            apply: pure(type_SomeObject),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — `@op.issubtype.register(SomeTypeOf)`.
    register(
        reg,
        OpKind::IsSubtype,
        SomeValueTag::TypeOf,
        Specialization {
            apply: pure(issubtype_SomeTypeOf),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py:91-111 — @op.contains.register(SomeObject|SomeNone|…)

#[allow(non_snake_case)]
pub fn contains_SomeObject(_ann: &RPythonAnnotator, _hl: &HLOperation) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(non_snake_case)]
pub fn contains_SomeNone(_ann: &RPythonAnnotator, _hl: &HLOperation) -> SomeValue {
    let mut s = SomeBool::new();
    s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
    SomeValue::Bool(s)
}

pub fn contains_number(_ann: &RPythonAnnotator, _hl: &HLOperation) -> SomeValue {
    panic!("AnnotatorError: number is not iterable")
}

fn init_contains_register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:91-94 — SomeObject: returns s_Bool, can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::Object,
        Specialization {
            apply: pure(contains_SomeObject),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:96-104 — SomeNone: returns SomeBool(const=False).
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::None_,
        Specialization {
            apply: pure(contains_SomeNone),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:107-111 — SomeInteger/Float/Bool: raise AnnotatorError.
    let number_tags = &[
        SomeValueTag::Integer,
        SomeValueTag::Float,
        SomeValueTag::Bool,
    ];
    for tag in number_tags {
        register(
            reg,
            OpKind::Contains,
            *tag,
            Specialization {
                apply: pure(contains_number),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// unaryop.py, 147-153, 1023-1027 — isinstance / issubtype

enum TypeDesc {
    Builtin(BuiltinTypeDesc),
    Class(std::rc::Rc<std::cell::RefCell<ClassDesc>>),
}

fn known_type_host(known: KnownType) -> Option<HostObject> {
    let name = match known {
        KnownType::Object => "object",
        KnownType::Type => "type",
        KnownType::Int => "int",
        KnownType::Bool => "bool",
        KnownType::Float => "float",
        KnownType::Str => "str",
        KnownType::Unicode => "unicode",
        KnownType::Bytearray => "bytearray",
        KnownType::NoneType => "NoneType",
        KnownType::BuiltinFunctionOrMethod => "builtin_function_or_method",
        KnownType::PropertyType => "property",
        KnownType::List => "list",
        KnownType::Tuple => "tuple",
        KnownType::Dict => "dict",
        KnownType::OrderedDict => "OrderedDict",
        KnownType::FunctionType => "function",
        KnownType::MethodType => "method",
        _ => return None,
    };
    HOST_ENV.lookup_builtin(name)
}

fn annotation_type_host(s_obj: &SomeValue) -> Option<HostObject> {
    match s_obj {
        SomeValue::Instance(s_inst) => s_inst
            .classdef
            .as_ref()
            .map(|cdef| cdef.borrow().classdesc.borrow().pyobj.clone()),
        _ => known_type_host(s_obj.knowntype()),
    }
}

fn const_type_host(s_type: &SomeValue) -> Option<&HostObject> {
    match s_type.const_() {
        Some(ConstValue::HostObject(host)) if host.is_class() => Some(host),
        _ => None,
    }
}

fn toclassdesc(
    bk: &std::rc::Rc<super::bookkeeper::Bookkeeper>,
    cls: &HostObject,
) -> Result<TypeDesc, AnnotatorError> {
    if is_primitive_type(cls) {
        return Ok(TypeDesc::Builtin(BuiltinTypeDesc::new(cls.clone())));
    }
    match bk.getdesc(cls)? {
        super::description::DescEntry::Class(classdesc) => Ok(TypeDesc::Class(classdesc)),
        other => Err(AnnotatorError::new(format!(
            "our_issubclass({:?}): expected ClassDesc, got {:?}",
            cls.qualname(),
            other.kind()
        ))),
    }
}

fn known_type_issubclass(cls1: KnownType, cls2: KnownType) -> bool {
    if cls1 == cls2 {
        return true;
    }
    matches!(
        (cls1, cls2),
        (KnownType::Bool, KnownType::Int) | (_, KnownType::Object)
    )
}

fn host_issubclass(
    bk: &std::rc::Rc<super::bookkeeper::Bookkeeper>,
    cls1: &HostObject,
    cls2: &HostObject,
) -> bool {
    match (
        known_type_host_from_host(cls1),
        known_type_host_from_host(cls2),
    ) {
        (Some(k1), Some(k2)) => known_type_issubclass(k1, k2),
        _ => our_issubclass(bk, cls1, cls2),
    }
}

pub fn our_issubclass(
    bk: &std::rc::Rc<super::bookkeeper::Bookkeeper>,
    cls1: &HostObject,
    cls2: &HostObject,
) -> bool {
    match (toclassdesc(bk, cls1), toclassdesc(bk, cls2)) {
        (Ok(TypeDesc::Builtin(desc1)), Ok(TypeDesc::Builtin(desc2))) => desc1.issubclass(&desc2),
        (Ok(TypeDesc::Class(desc1)), Ok(TypeDesc::Class(desc2))) => {
            desc1.borrow().issubclass(&desc2.borrow())
        }
        _ => cls1.is_subclass_of(cls2),
    }
}

pub fn s_isinstance(
    annotator: &RPythonAnnotator,
    s_obj: &SomeValue,
    s_type: &SomeValue,
    variables: Vec<Rc<super::super::flowspace::model::Variable>>,
) -> SomeValue {
    if !s_type.is_constant() {
        return SomeValue::Bool(SomeBool::new());
    }
    let Some(typ) = const_type_host(s_type) else {
        return SomeValue::Bool(SomeBool::new());
    };
    let mut r = SomeBool::new();
    let bk = &annotator.bookkeeper;
    if let Some(obj_const) = s_obj.const_() {
        if let Some(obj_type) = obj_const.class_of() {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(
                obj_type.is_subclass_of(typ),
            )));
        }
    } else if let Some(obj_type) = annotation_type_host(s_obj) {
        if host_issubclass(bk, &obj_type, typ) {
            if !s_obj.can_be_none() {
                r.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
            }
        } else if !host_issubclass(bk, typ, &obj_type)
            || (s_obj.knowntype() == KnownType::Int
                && known_type_host(KnownType::Bool).as_ref() == Some(typ))
        {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        }
    }
    for v in &variables {
        let binding = annotator
            .annotation(&Hlvalue::Variable((**v).clone()))
            .expect("s_isinstance: variable has no binding");
        assert_eq!(&binding, s_obj);
    }
    let mut knowntypedata = std::collections::HashMap::new();
    if !typ.class_has("_freeze_")
        && matches!(
            s_type,
            SomeValue::PBC(pbc) if pbc.get_kind().ok() == Some(super::model::DescKind::Class)
        )
        && let Some(spec) = annotation_spec_for_host_type(typ)
        && let Ok(s_value) = bk.valueoftype(&spec)
    {
        add_knowntypedata(
            &mut knowntypedata,
            ExitCaseKey::Bool(true),
            &variables,
            s_value,
        );
    }
    r.set_knowntypedata(knowntypedata);
    SomeValue::Bool(r)
}

fn known_type_host_from_host(host: &HostObject) -> Option<KnownType> {
    match host.qualname() {
        "object" => Some(KnownType::Object),
        "type" => Some(KnownType::Type),
        "int" => Some(KnownType::Int),
        "bool" => Some(KnownType::Bool),
        "float" => Some(KnownType::Float),
        "str" => Some(KnownType::Str),
        "unicode" => Some(KnownType::Unicode),
        "bytearray" => Some(KnownType::Bytearray),
        "NoneType" => Some(KnownType::NoneType),
        "builtin_function_or_method" => Some(KnownType::BuiltinFunctionOrMethod),
        "property" => Some(KnownType::PropertyType),
        _ => None,
    }
}

fn annotation_spec_for_host_type(host: &HostObject) -> Option<super::signature::AnnotationSpec> {
    match host.qualname() {
        "bool" => Some(super::signature::AnnotationSpec::Bool),
        "int" => Some(super::signature::AnnotationSpec::Int),
        "float" => Some(super::signature::AnnotationSpec::Float),
        "str" => Some(super::signature::AnnotationSpec::Str),
        "unicode" => Some(super::signature::AnnotationSpec::Unicode),
        "NoneType" => Some(super::signature::AnnotationSpec::NoneType),
        "type" => Some(super::signature::AnnotationSpec::Type),
        _ if host.is_class() => Some(super::signature::AnnotationSpec::UserClass(host.clone())),
        _ => None,
    }
}

#[allow(non_snake_case)]
pub fn isinstance_SomeObject(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_obj = ann
        .annotation(&hl.args[0])
        .expect("isinstance: object unbound");
    let s_cls = ann
        .annotation(&hl.args[1])
        .expect("isinstance: cls unbound");
    let variables = match &hl.args[0] {
        Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
        Hlvalue::Constant(_) => Vec::new(),
    };
    s_isinstance(ann, &s_obj, &s_cls, variables)
}

pub fn issubtype(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_type = ann
        .annotation(&hl.args[0])
        .expect("issubtype: type unbound");
    let s_cls = ann.annotation(&hl.args[1]).expect("issubtype: cls unbound");
    if s_type.is_constant()
        && s_cls.is_constant()
        && let (Some(type_host), Some(cls_host)) =
            (const_type_host(&s_type), const_type_host(&s_cls))
    {
        return ann
            .bookkeeper
            .immutablevalue(&ConstValue::Bool(type_host.is_subclass_of(cls_host)))
            .expect("immutablevalue(bool) must succeed");
    }
    s_bool()
}

#[allow(non_snake_case)]
fn issubtype_SomeTypeOf(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let Some(SomeValue::TypeOf(type_of)) = ann.annotation(&hl.args[0]) else {
        panic!("issubtype(SomeTypeOf): arg 0 is not SomeTypeOf");
    };
    let Some(first) = type_of.is_type_of.first() else {
        return SomeValue::Bool(SomeBool::new());
    };
    let Some(s_obj) = ann.annotation(&Hlvalue::Variable((**first).clone())) else {
        return SomeValue::Bool(SomeBool::new());
    };
    let s_cls = ann.annotation(&hl.args[1]).expect("issubtype: cls unbound");
    s_isinstance(ann, &s_obj, &s_cls, type_of.is_type_of.to_vec())
}

// unaryop.py — class __extend__(SomeObject) simple-return defaults
//
// `class __extend__(SomeObject)` upstream defines the defaults fetched by
// `SingleDispatchMixin.get_specialization` when `getattr(s_arg, opname)`
// hits the base class. Rust port registers one spec per opname on
// `SomeValueTag::Object`; the MRO fallback handles every subclass that
// doesn't override.

#[allow(non_snake_case)]
pub fn bool_SomeObject(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_obj = ann.annotation(&hl.args[0]).expect("bool: object unbound");
    let mut r = SomeBool::new();
    // Dispatch of `annotation(obj).bool_behavior(r)` over the per-type
    // `bool_behavior` overrides.
    match &s_obj {
        // unaryop.py `SomeInstance.bool_behavior`.
        SomeValue::Instance(s_inst) if !s_inst.can_be_none => {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
        }
        // unaryop.py `SomePBC.bool_behavior`.
        SomeValue::PBC(pbc) if !pbc.can_be_none => {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(true)));
        }
        // unaryop.py `SomeNone.bool_behavior`.
        SomeValue::None_(_) => {
            r.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        }
        // unaryop.py `SomeObject.bool_behavior`.
        _ => {
            if s_obj.is_immutable_constant()
                && let Some(c) = s_obj.const_()
                && let Some(truthy) = c.truthy()
            {
                r.base.const_box = Some(Constant::new(ConstValue::Bool(truthy)));
            } else {
                let len_op = HLOperation::new(OpKind::Len, vec![hl.args[0].clone()]);
                let s_len = len_op
                    .consider(ann)
                    .expect("bool: len() dispatch failed")
                    .expect("bool: len() is not a void operation");
                if let Some(n) = s_len.const_().and_then(ConstValue::as_int) {
                    r.base.const_box = Some(Constant::new(ConstValue::Bool(n > 0)));
                }
            }
        }
    }
    let mut s_nonnone_obj = s_obj.clone();
    if s_nonnone_obj.can_be_none() {
        s_nonnone_obj = s_nonnone_obj.nonnoneify();
    }
    let vars = match &hl.args[0] {
        Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
        Hlvalue::Constant(_) => Vec::new(),
    };
    let mut knowntypedata = std::collections::HashMap::new();
    add_knowntypedata(
        &mut knowntypedata,
        ExitCaseKey::Bool(true),
        &vars,
        s_nonnone_obj,
    );
    r.set_knowntypedata(knowntypedata);
    SomeValue::Bool(r)
}

/// `bool(disc)` of an enum `__discriminant` read (`SomeFloat.bool`, which
/// `SomeInteger` inherits) keeps its per-tag variant narrowing, the way
/// `SomeBool.bool` returns `self` and keeps its `knowntypedata`
/// (unaryop.py): the `False` arm is tag `0`, the
/// `True` arm every other tag, refined to the union of their variants
/// (`merge_knowntypedata`).  The tag-keyed table is pyre's own (enum
/// discriminant narrowing, see `SomeInstance.getattr("__discriminant")`);
/// without this a two-variant `match` lowered to `set_branch` reads the
/// variant payload off the unrefined base class.
fn bool_of_discriminant_knowntypedata(
    discriminant_ktd: &super::model::KnownTypeData,
) -> super::model::KnownTypeData {
    let mut knowntypedata = super::model::KnownTypeData::new();
    for (case, constraints) in discriminant_ktd {
        let ExitCaseKey::Int(tag) = case else {
            continue;
        };
        let key = ExitCaseKey::Bool(*tag != 0);
        let mut arm = super::model::KnownTypeData::new();
        arm.insert(key, constraints.clone());
        if *tag != 0 && knowntypedata.contains_key(&key) {
            arm = super::model::merge_knowntypedata(&knowntypedata, &arm);
        }
        knowntypedata.extend(arm);
    }
    knowntypedata
}

#[allow(non_snake_case)]
pub fn simple_call_SomeObject(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    // Mirror RPython `unaryop.py:114-118`:
    //
    //     s_func = annotator.annotation(func)
    //     argspec = simple_args([annotator.annotation(arg) for arg in args])
    //     return s_func.call(argspec)
    //
    // `annotator.annotation(arg)` returns `None` when the variable is
    // not yet bound (`annrpython.py`); pyre's
    // `arguments_w: Vec<Option<SomeValue>>` carries the None through
    // to consumers, matching `bookkeeper.py consider_call_site` parity. When
    // `s_func` itself is unbound, the call is deferred (return
    // Impossible so the fixpoint propagates Bottom upward; the next
    // round re-fires once the callable's annotation populates).
    let Some(s_func) = ann.annotation(&hl.args[0]) else {
        return Some(super::model::s_impossible_value());
    };
    let args_s: Vec<Option<SomeValue>> = hl.args[1..].iter().map(|a| ann.annotation(a)).collect();
    let argspec = super::argument::simple_args_opt(args_s);
    // `s_func.call` returns `Ok(None)` for a void analyser (a builtin
    // method body that falls off the end); the None threads straight
    // through to `consider_op`, which binds Impossible without blocking.
    match s_func.call(&argspec) {
        Ok(cell) => cell,
        // `compute_at_fixpoint` (bookkeeper.py) propagates uncaught, so the
        // error leaves the annotator as itself. Carrying the
        // `AnnotatorError` rather than a rendering of it keeps the failing
        // stage in the payload's type, which is how `flowin`
        // (annrpython.py) knows to attach `source` and how `gottypererror`
        // (rtyper.py) stays a separate handler; `in_stage`
        // names the entry point inside the message the way upstream
        // composes it at the raise site.
        Err(e) => std::panic::panic_any(e.in_stage("compute_at_fixpoint")),
    }
}

pub fn call_args(ann: &RPythonAnnotator, hl: &HLOperation) -> Option<SomeValue> {
    let Some(s_func) = ann.annotation(&hl.args[0]) else {
        return Some(super::model::s_impossible_value());
    };
    let Some(Hlvalue::Constant(shape_const)) = hl.args.get(1) else {
        panic!("AnnotatorError: call_args shape must be constant");
    };
    let shape = decode_call_shape(&shape_const.value)
        .unwrap_or_else(|| panic!("AnnotatorError: invalid call_args shape"));
    let args_s: Vec<Option<SomeValue>> =
        hl.args[2..].iter().map(|arg| ann.annotation(arg)).collect();
    let callspec = super::argument::complex_args(&shape, args_s);
    match s_func.call(&callspec) {
        Ok(cell) => cell,
        Err(e) => std::panic::panic_any(e.in_stage("compute_at_fixpoint")),
    }
}

fn init_someobject_defaults(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — `@op.isinstance.register(SomeObject)`.
    register(
        reg,
        OpKind::IsInstance,
        SomeValueTag::Object,
        Specialization {
            apply: pure(isinstance_SomeObject),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — `@op.issubtype.register(SomeObject)`.
    register(
        reg,
        OpKind::IsSubtype,
        SomeValueTag::Object,
        Specialization {
            apply: pure(issubtype),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — len(self): return SomeInteger(nonneg=True)
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py / 151-156 — bool(self): bool_behavior +
    // truth-branch knowntypedata via nonnoneify().
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Object,
        Specialization {
            apply: pure(bool_SomeObject),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — hash: raise AnnotatorError("cannot use hash() in RPython").
    register(
        reg,
        OpKind::Hash,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| panic!("AnnotatorError: cannot use hash() in RPython")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — str: SomeString().
    register(
        reg,
        OpKind::Str,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — repr: SomeString().
    register(
        reg,
        OpKind::Repr,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — hex/bin/oct: SomeString().
    for op in &[OpKind::Hex, OpKind::Bin, OpKind::Oct] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: pure(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py — id: raise AnnotatorError.
    register(
        reg,
        OpKind::Id,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| {
                panic!("AnnotatorError: cannot use id() in RPython; see objectmodel.compute_xxx()")
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — int: SomeInteger().
    register(
        reg,
        OpKind::Int,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — float: SomeFloat().
    register(
        reg,
        OpKind::Float,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — SomeObject.getattr(self, s_attr).
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|ann, hl| {
                let s_self = ann.annotation(&hl.args[0]).expect("getattr: self unbound");
                let s_attr = ann.annotation(&hl.args[1]).expect("getattr: attr unbound");
                // unaryop.py:280-285 `_getattr_(s_obj, s_attr)` — `if
                // s_attr.is_constant() and isinstance(s_attr.const, str)`.
                // Upstream Python 2 `str` is bytes; pyre uses
                // [`ConstValue::as_pystr`] so a unicode literal does not
                // sneak through.
                let Some(attr) = s_attr
                    .const_()
                    .and_then(ConstValue::as_pystr)
                    .map(str::to_owned)
                else {
                    panic!(
                        "AnnotatorError: getattr({:?}, {:?}) has non-constant argument",
                        s_self, s_attr
                    );
                };
                if let Some(s_method) = s_self.find_method(&attr) {
                    return s_method;
                }
                // upstream unaryop.py getattr constant receiver path
                // shares the `getattr(obj, name)` call with
                // `flowspace::operation::GetAttr.constfold`; both
                // route through `const_runtime_getattr` and only
                // diverge at the wrap step (annotator wraps via
                // `immutablevalue`, flowspace via `const`).
                if s_self.is_immutable_constant()
                    && let Some(c) = s_self.const_()
                {
                    match crate::flowspace::model::const_runtime_getattr(c, &attr) {
                        Ok(Some(value)) => {
                            return ann.bookkeeper.immutablevalue(&value).unwrap_or_else(|err| {
                                panic!("getattr immutablevalue failed: {}", err)
                            });
                        }
                        Ok(None) => {
                            // upstream `WrapException` swallow — fall
                            // through to the "cannot find attr" panic.
                        }
                        Err(msg) => {
                            panic!("AnnotatorError in getattr constant path: {msg}")
                        }
                    }
                }
                // A raw-pointer `p.is_null()` whose receiver Charon erases to
                // a pointer-to-collection (`*mut Vec<T>` → `SomeList`,
                // `*mut DictStorage` → `SomeDict`) reaches this `SomeObject`
                // base getattr rather than the `SomeInstance` `is_null` arm.
                // `front::mir` already routed it as `CallTarget::Method {
                // name: "is_null", receiver_root: "mut_ptr"/"const_ptr" }`,
                // which `jtransform` lowers to `ptr_iszero` regardless of the
                // receiver repr; answer the getattr with the same
                // `ptr_method_is_null` bound method so the result types as
                // `SomeBool`.  Last-resort (after `find_method` + the constant
                // path), so a real `is_null` member is never shadowed.
                if attr == "is_null"
                    && matches!(
                        s_self,
                        SomeValue::List(_) | SomeValue::Dict(_) | SomeValue::RustVec(_)
                    )
                {
                    return SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
                        "ptr_method_is_null",
                        s_self.clone(),
                        "is_null",
                    ));
                }
                // A Rust tuple field is `__pos_N`. The value is already a
                // `SomeTuple`; the item annotation is that slot.
                if let SomeValue::Tuple(ref t) = s_self
                    && let Some(rest) = attr.strip_prefix("__pos_")
                    && let Ok(idx) = rest.parse::<usize>()
                    && let Some(item) = t.items.get(idx)
                {
                    return item.clone();
                }
                panic!(
                    "AnnotatorError: Cannot find attribute {:?} on {:?}",
                    attr, s_self
                )
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — SomePBC.getattr(self, s_attr).
    //
    //   def getattr(self, s_attr):
    //       assert s_attr.is_constant()
    //       if s_attr.const == '__name__':
    //           from rpython.annotator.classdesc import ClassDesc
    //           if self.getKind() is ClassDesc:
    //               return SomeString()
    //       bookkeeper = getbookkeeper()
    //       return bookkeeper.pbc_getattr(self, s_attr)
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::PBC,
        Specialization {
            apply: pure(|ann, hl| {
                let s_self = ann
                    .annotation(&hl.args[0])
                    .expect("pbc getattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("pbc getattr: attr unbound");
                let SomeValue::PBC(pbc) = &s_self else {
                    unreachable!("PBC-tag dispatch must carry SomePBC, got {s_self:?}");
                };
                // unaryop.py — `SomePBC.getattr(self, s_attr)`.
                // Upstream only asserts `s_attr.is_constant()` and
                // uses `s_attr.const` directly — no `isinstance(...,
                // str)` gate. Python 2 string equality crosses the
                // bytes/unicode boundary (`'__name__' == u'__name__'`
                // is True), so accept both `ByteStr` and `UniStr` via
                // [`ConstValue::as_text`] to keep parity with the
                // upstream constant-only check.
                let Some(attr) = s_attr
                    .const_()
                    .and_then(ConstValue::as_text)
                    .map(str::to_owned)
                else {
                    panic!(
                        "AnnotatorError: pbc getattr({:?}, {:?}) has non-constant argument",
                        s_self, s_attr
                    );
                };
                // __name__ special-case: ClassDesc PBC → SomeString.
                if attr == "__name__" && matches!(pbc.get_kind(), Ok(super::model::DescKind::Class))
                {
                    return SomeValue::String(SomeString::new(false, false));
                }
                match ann.bookkeeper.pbc_getattr(pbc, &s_attr) {
                    Ok(s) => s,
                    Err(super::model::AnnotatorException::Annotator(e)) => {
                        panic!("AnnotatorError in pbc_getattr: {e}")
                    }
                    Err(super::model::AnnotatorException::Harmless(_)) => {
                        // upstream lets `HarmlesslyBlocked` propagate out of
                        // flowin; the Rust port has no exception channel
                        // across Specialization::apply yet, so surface as
                        // Impossible and let the reflow loop retry once the
                        // enforced attr family widens.
                        super::model::s_impossible_value()
                    }
                }
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — delattr: warning-only. Returns no value
    // (upstream implicit None). Port as a void specialization returning
    // `None`, so consider binds the result to Impossible without blocking.
    register(
        reg,
        OpKind::DelAttr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| None),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — `@op.simple_call.register(SomeObject)`.
    //
    //     @op.simple_call.register(SomeObject)
    //     def simple_call_SomeObject(annotator, func, *args):
    //         s_func = annotator.annotation(func)
    //         argspec = simple_args([annotator.annotation(arg) for arg in args])
    //         return s_func.call(argspec)
    //
    // Polymorphic dispatch entry — registered against `SomeObject` so
    // every `s_func` tag (`SomePBC`, `SomeBuiltinMethod`, `SomePtr`,
    // …) inherits this specialization via mro lookup.  Each subtype's
    // `.call(argspec)` then routes to the type-specific handler:
    // `SomePBC.call → Bookkeeper.pbc_call → FunctionDesc.pycall →
    // annotator.recursivecall → addpendingblock`
    // (`description.py`, `annrpython.py`).
    register(
        reg,
        OpKind::SimpleCall,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(simple_call_SomeObject),
            // upstream `simple_call.can_only_throw` is unset →
            // `CanOnlyThrow::Absent` matches `_can_only_throw_None`'s
            // path for ops without an explicit static throw set.
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — `@op.call_args.register(SomeObject)`.
    register(
        reg,
        OpKind::CallArgs,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(call_args),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — hint(self, *args_s): return self.
    register(
        reg,
        OpKind::Hint,
        SomeValueTag::Object,
        Specialization {
            apply: pure(|ann, hl| ann.annotation(&hl.args[0]).expect("hint: self unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — getslice/setslice/delslice: s_ImpossibleValue.
    for op in &[OpKind::GetSlice, OpKind::SetSlice, OpKind::DelSlice] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: pure(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py — pos/neg/abs/ord/invert/long/iter/next: s_ImpossibleValue.
    //
    //     def pos(self):
    //         return s_ImpossibleValue
    //     neg = abs = ord = invert = long = iter = next = pos
    for op in &[
        OpKind::Pos,
        OpKind::Neg,
        OpKind::Abs,
        OpKind::Ord,
        OpKind::Invert,
        OpKind::Long,
        OpKind::Iter,
        OpKind::Next,
    ] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: pure(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// unaryop.py — class __extend__(SomeFloat)

fn init_somefloat_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — pos(self): return self.
    register(
        reg,
        OpKind::Pos,
        SomeValueTag::Float,
        Specialization {
            apply: pure(|ann, hl| ann.annotation(&hl.args[0]).expect("float.pos: unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — neg(self): SomeFloat(). abs = neg.
    for op in &[OpKind::Neg, OpKind::Abs] {
        register(
            reg,
            *op,
            SomeValueTag::Float,
            Specialization {
                apply: pure(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py — bool(self): SomeBool (const if self is constant).
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Float,
        Specialization {
            apply: pure(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("float.bool: unbound");
                if let SomeValue::Float(f) = &s
                    && f.is_immutable_constant()
                    && let Some(c) = &f.base.const_box
                    && let ConstValue::Float(bits) = c.value
                {
                    let mut r = SomeBool::new();
                    r.base.const_box =
                        Some(Constant::new(ConstValue::Bool(f64::from_bits(bits) != 0.0)));
                    return SomeValue::Bool(r);
                }
                let mut r = SomeBool::new();
                if let SomeValue::Integer(s_int) = &s
                    && let Some(discriminant_ktd) = &s_int.knowntypedata
                {
                    r.set_knowntypedata(bool_of_discriminant_knowntypedata(discriminant_ktd));
                }
                SomeValue::Bool(r)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — len(self): raise AnnotatorError.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Float,
        Specialization {
            apply: pure(|_ann, _hl| panic!("AnnotatorError: 'float' has no length")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeInteger)

fn init_someinteger_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — invert/pos/int: SomeInteger(knowntype=self.knowntype).
    //                      can_only_throw = [].
    let same_kt_int = |ann: &RPythonAnnotator, hl: &HLOperation| -> SomeValue {
        let kt = match ann.annotation(&hl.args[0]) {
            Some(SomeValue::Integer(i)) => i.base.knowntype,
            _ => panic!("integer op: arg 0 not SomeInteger"),
        };
        SomeValue::Integer(SomeInteger::new_with_knowntype(false, kt))
    };
    for op in &[OpKind::Invert, OpKind::Pos, OpKind::Int] {
        register(
            reg,
            *op,
            SomeValueTag::Integer,
            Specialization {
                apply: pure(same_kt_int),
                can_only_throw: CanOnlyThrow::List(vec![]),
            },
        );
    }
    // unaryop.py — neg: SomeInteger(knowntype=self.knowntype). neg_ovf overflows.
    register(
        reg,
        OpKind::Neg,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(same_kt_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::NegOvf,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(same_kt_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py — abs: SomeInteger(nonneg=True, knowntype=self.knowntype).
    let abs_int = |ann: &RPythonAnnotator, hl: &HLOperation| -> SomeValue {
        let kt = match ann.annotation(&hl.args[0]) {
            Some(SomeValue::Integer(i)) => i.base.knowntype,
            _ => panic!("integer.abs: arg 0 not SomeInteger"),
        };
        // upstream: SomeInteger(nonneg=True, knowntype=self.knowntype).
        let mut i = SomeInteger::new_with_knowntype(true, kt);
        i.nonneg = true;
        SomeValue::Integer(i)
    };
    register(
        reg,
        OpKind::Abs,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(abs_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::AbsOvf,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(abs_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py — len: raise.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Integer,
        Specialization {
            apply: pure(|_ann, _hl| panic!("AnnotatorError: 'int' has no length")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeBool)

fn init_somebool_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — bool(self): return self.
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(|ann, hl| ann.annotation(&hl.args[0]).expect("bool.bool: unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — invert: SomeInteger(). can_only_throw=[].
    register(
        reg,
        OpKind::Invert,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — neg: SomeInteger(). neg_ovf overflows.
    register(
        reg,
        OpKind::Neg,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::NegOvf,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py — abs / abs_ovf / pos / int: SomeInteger(nonneg=True).
    let nonneg_int = |_ann: &RPythonAnnotator, _hl: &HLOperation| -> SomeValue {
        let i = SomeInteger {
            nonneg: true,
            ..SomeInteger::default()
        };
        SomeValue::Integer(i)
    };
    register(
        reg,
        OpKind::Abs,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::AbsOvf,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    register(
        reg,
        OpKind::Pos,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::Int,
        SomeValueTag::Bool,
        Specialization {
            apply: pure(nonneg_int),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeTuple)

fn init_sometuple_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — len(self): immutablevalue(len(self.items)).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Tuple,
        Specialization {
            apply: pure(|ann, hl| {
                let t = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Tuple(t)) => t,
                    _ => panic!("tuple.len: arg 0 not SomeTuple"),
                };
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(t.items.len() as i64)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Tuple,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("tuple.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — getslice((self, s_start, s_stop)): require
    // constant indices, slice items, rebuild SomeTuple. Registered as
    // the primary SomeTuple.getslice handler (the SomeObject fallback
    // would have returned s_ImpossibleValue).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::Tuple,
        Specialization {
            apply: pure(|ann, hl| {
                let t = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Tuple(t)) => t,
                    _ => panic!("tuple.getslice: arg 0 not SomeTuple"),
                };
                let start = ann
                    .annotation(&hl.args[1])
                    .expect("tuple.getslice: start unbound");
                let stop = ann
                    .annotation(&hl.args[2])
                    .expect("tuple.getslice: stop unbound");
                let slice_bound = |sv: &SomeValue, which: &str| -> usize {
                    let i = match sv {
                        SomeValue::Integer(i) if i.is_immutable_constant() => i,
                        _ => panic!(
                            "AnnotatorError: tuple slicing: needs constants ({which} not const)"
                        ),
                    };
                    let c = i
                        .base
                        .const_box
                        .as_ref()
                        .expect("tuple.getslice: const_box mismatch");
                    let len = t.items.len();
                    match &c.value {
                        ConstValue::Int(v) => {
                            if *v < 0 {
                                (len as i64 + *v).max(0) as usize
                            } else {
                                (*v as usize).min(len)
                            }
                        }
                        // `SomeTuple.getslice` indexes the Python list with
                        // the actual positive `r_uint` value; a huge bound
                        // clamps to `len`, never wrapping negative.
                        ConstValue::UInt(v) => {
                            if (*v as u128) >= len as u128 {
                                len
                            } else {
                                *v as usize
                            }
                        }
                        _ => panic!("tuple.getslice: const_box mismatch"),
                    }
                };
                let lo = slice_bound(&start, "start");
                let hi = slice_bound(&stop, "stop").max(lo);
                let items = t.items[lo..hi].to_vec();
                SomeValue::Tuple(SomeTuple::new(items))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — @op.contains.register(SomeList) + class __extend__(SomeList)

#[allow(non_snake_case)]
pub fn contains_SomeList(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_list = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::List(s)) => s,
        _ => panic!("contains(SomeList): arg 0 not SomeList"),
    };
    let s_element = ann
        .annotation(&hl.args[1])
        .expect("contains(SomeList): element unbound");
    s_list
        .listdef
        .generalize(&s_element)
        .expect("contains.SomeList: generalize failed");
    SomeValue::Bool(SomeBool::new())
}

/// `len(v)` on a Rust `Vec`: a non-negative integer.
fn init_somerustvec_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    register(
        reg,
        OpKind::Len,
        SomeValueTag::RustVec,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::RustVec,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("rustvec.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // `SomePtr.getattr` (lltype.py) on the raw `{ptr, len, cap}` header.
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::RustVec,
        Specialization {
            apply: pure(|ann, hl| {
                let s = ann
                    .annotation(&hl.args[0])
                    .expect("rustvec.getattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("rustvec.getattr: attr unbound");
                let SomeValue::RustVec(s_vec) = s else {
                    unreachable!("RustVec-tag dispatch must carry SomeRustVec, got {s:?}");
                };
                s_vec
                    .getattr(&s_attr)
                    .unwrap_or_else(|err| std::panic::panic_any(err))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // `SomePtr.setattr` (lltype.py) — "just doing checking", void.
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::RustVec,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann
                    .annotation(&hl.args[0])
                    .expect("rustvec.setattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("rustvec.setattr: attr unbound");
                let s_value = ann
                    .annotation(&hl.args[2])
                    .expect("rustvec.setattr: value unbound");
                let SomeValue::RustVec(s_vec) = s else {
                    unreachable!("RustVec-tag dispatch must carry SomeRustVec, got {s:?}");
                };
                s_vec
                    .setattr(&s_attr, &s_value)
                    .unwrap_or_else(|err| std::panic::panic_any(err));
                None
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py `SomeList.getslice`: check_negative_slice + listdef.offspring.
    // SomeRustVec has a fixed s_item and no listdef, so the offspring analog
    // is a fresh SomeRustVec of the same item.
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::RustVec,
        Specialization {
            apply: pure(|ann, hl| {
                let s_vec = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::RustVec(s)) => s,
                    _ => panic!("rustvec.getslice: arg 0 not SomeRustVec"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("rustvec.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("rustvec.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::RustVec(SomeRustVec::new((*s_vec.s_item).clone()))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_somelist_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:350-354 — contains: generalize + s_Bool, can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::List,
        Specialization {
            apply: pure(contains_SomeList),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — len(self): position read_item → 0 if impossible,
    // else SomeObject.len.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::List,
        Specialization {
            apply: pure(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.len: arg 0 not SomeList"),
                };
                let position = ann.bookkeeper.current_position_key();
                let s_item = s_list.listdef.read_item(position);
                if matches!(s_item, SomeValue::Impossible) {
                    // upstream: immutablevalue(0) → SomeInteger with const=0.
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                    SomeValue::Integer(i)
                } else {
                    // upstream: SomeObject.len(self) → SomeInteger(nonneg=True).
                    SomeValue::Integer(SomeInteger::new(true, false))
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::List,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("list.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — getslice((self, s_start, s_stop)):
    //     check_negative_slice(...); return listdef.offspring(bk).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::List,
        Specialization {
            apply: pure(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.getslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                let res = s_list
                    .listdef
                    .offspring(&ann.bookkeeper, &[])
                    .expect("listdef.offspring failed");
                SomeValue::List(res)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — setslice((self, s_start, s_stop), s_iterable):
    //     check_negative + isinstance(SomeList) + mutate + agree + resize.
    //     No return → None (void).
    register(
        reg,
        OpKind::SetSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.setslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.setslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.setslice: stop unbound");
                let s_iterable = ann
                    .annotation(&hl.args[3])
                    .expect("list.setslice: iterable unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                let s_other = match s_iterable {
                    SomeValue::List(o) => o,
                    _ => panic!("AnnotatorError: list[start:stop] = x: x must be a list"),
                };
                s_list.listdef.mutate().expect("listdef.mutate failed");
                s_list
                    .listdef
                    .agree(&ann.bookkeeper, &s_other.listdef)
                    .expect("listdef.agree failed");
                s_list.listdef.resize().expect("listdef.resize failed");
                None
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — delslice((self, s_start, s_stop)):
    //     check_negative + resize. No return → None (void).
    register(
        reg,
        OpKind::DelSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.delslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.delslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.delslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                s_list.listdef.resize().expect("listdef.resize failed");
                None
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

/// RPython `check_negative_slice(s_start, s_stop, error="slicing")`
/// (unaryop.py). Helper shared by list and string slice paths.
pub fn check_negative_slice(s_start: &SomeValue, s_stop: &SomeValue, error: &str) {
    if let SomeValue::Integer(i) = s_start
        && !i.nonneg
    {
        panic!("AnnotatorError: {error}: not proven to have non-negative start");
    }
    if let SomeValue::Integer(i) = s_stop
        && !i.nonneg
    {
        // upstream: allow const == -1 as sentinel for "slice to end".
        let is_minus_one = i
            .base
            .const_box
            .as_ref()
            .map(|c| matches!(c.value, ConstValue::Int(-1)))
            .unwrap_or(false);
        if !is_minus_one {
            panic!("AnnotatorError: {error}: not proven to have non-negative stop");
        }
    }
}

// unaryop.py __extend__ — SomeList.method_* free functions
//
// Upstream these are class methods on `class __extend__(SomeList)`. They
// are NOT directly `@op.X.register` targets; instead `SomeObject.getattr`
// looks up `method_<name>` via `find_method` and wraps them in a
// `SomeBuiltinMethod` for downstream `simple_call`. The Rust port keeps
// the analyzer bodies as free functions and the `find_method` /
// `SomeBuiltinMethod.call` bridge below dispatches to them.

#[allow(dead_code)]
fn list_method_append(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py — method_append falls off the end (returns None
    // = void op), so the call binds Impossible without blocking.
    s_self.listdef.resize().expect("resize");
    s_self.listdef.generalize(s_value).expect("generalize");
    None
}

#[allow(dead_code)]
fn list_method_extend(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_iterable: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:361-367 — falls off the end (void op).
    s_self.listdef.resize().expect("resize");
    if let SomeValue::List(other) = s_iterable {
        s_self
            .listdef
            .agree(&ann.bookkeeper, &other.listdef)
            .expect("agree");
    } else {
        // upstream:
        //     s_iter = s_iterable.iter()
        //     self.method_append(s_iter.next())
        //
        // `SomeIterator(s_iterable, variant=())` mirrors every
        // `SomeX.iter()` handler (unaryop.py/389/660/806/…),
        // which all reduce to `SomeIterator(self)` on the iterable.
        // `someiterator_next` is the extracted body of upstream
        // `SomeIterator.next` (unaryop.py).
        let s_iter = SomeIterator::new(s_iterable.clone(), vec![]);
        let s_item = someiterator_next(ann, &s_iter);
        list_method_append(ann, s_self, &s_item);
    }
    None
}

#[allow(dead_code)]
fn descr_reverse(_ann: &RPythonAnnotator, s_self: &super::model::SomeList) -> Option<SomeValue> {
    // unaryop.py:369-370 — falls off the end (void op).
    s_self.listdef.mutate().expect("listdef.mutate");
    None
}

#[allow(dead_code)]
fn descr_insert(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    _s_index: &SomeValue,
    s_value: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py — delegates to method_append (also void).
    list_method_append(ann, s_self, s_value)
}

#[allow(dead_code)]
fn descr_remove(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:375-377 — resize + generalize, falls off the end (void).
    s_self.listdef.resize().expect("resize");
    s_self.listdef.generalize(s_value).expect("generalize");
    None
}

#[allow(dead_code)]
fn list_method_pop(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    _s_index: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py method_pop — resize + read_item(position). can_only_throw=[IndexError].
    s_self.listdef.resize().expect("resize");
    let position = ann.bookkeeper.current_position_key();
    s_self.listdef.read_item(position)
}

#[allow(dead_code)]
fn list_method_index(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py method_index — generalize + SomeInteger(nonneg=True).
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Integer(SomeInteger::new(true, false))
}

// unaryop.py — @op.contains.register(SomeDict) + dict_contains helper

#[allow(non_snake_case)]
pub fn contains_SomeDict(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_dct = match ann.annotation(&hl.args[0]) {
        Some(SomeValue::Dict(d)) => d,
        _ => panic!("contains(SomeDict): arg 0 not SomeDict"),
    };
    let s_elem = ann
        .annotation(&hl.args[1])
        .expect("contains(SomeDict): element unbound");
    let position = ann.bookkeeper.current_position_key();
    dict_contains(&s_dct, &s_elem, position)
}

/// RPython `dict_contains(s_dct, s_element, position)` (unaryop.py).
///
/// ```python
/// def dict_contains(s_dct, s_element, position):
///     s_dct.dictdef.generalize_key(s_element)
///     if s_dct._is_empty(position):
///         s_bool = SomeBool(); s_bool.const = False; return s_bool
///     return s_Bool
/// ```
pub fn dict_contains(
    s_dct: &super::model::SomeDict,
    s_element: &SomeValue,
    position: Option<super::bookkeeper::PositionKey>,
) -> SomeValue {
    s_dct
        .dictdef
        .generalize_key(s_element)
        .expect("dict_contains: generalize_key failed");
    if dict_is_empty(s_dct, position) {
        let mut s = SomeBool::new();
        s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        return SomeValue::Bool(s);
    }
    SomeValue::Bool(SomeBool::new())
}

/// RPython `SomeDict._is_empty(self, position)` (unaryop.py).
fn dict_is_empty(
    s_dct: &super::model::SomeDict,
    position: Option<super::bookkeeper::PositionKey>,
) -> bool {
    let s_key = s_dct.dictdef.read_key(position.clone());
    let s_value = s_dct.dictdef.read_value(position);
    matches!(s_key, SomeValue::Impossible) || matches!(s_value, SomeValue::Impossible)
}

fn init_somedict_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:454-460 — @op.contains.register(SomeDict):
    //     position = annotator.bookkeeper.position_key
    //     return dict_contains(...); can_only_throw = _dict_can_only_throw_nothing.
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::Dict,
        Specialization {
            apply: pure(contains_SomeDict),
            can_only_throw: CanOnlyThrow::Callable(Box::new(|args_s| {
                // Mirror binaryop::_dict_can_only_throw_nothing (binaryop.py).
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
            })),
        },
    );
    // unaryop.py — len(self): 0 if empty, else SomeObject.len.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Dict,
        Specialization {
            apply: pure(|ann, hl| {
                let s_dct = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Dict(d)) => d,
                    _ => panic!("dict.len: arg 0 not SomeDict"),
                };
                let position = ann.bookkeeper.current_position_key();
                if dict_is_empty(&s_dct, position) {
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                    SomeValue::Integer(i)
                } else {
                    SomeValue::Integer(SomeInteger::new(true, false))
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Dict,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("dict.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

// unaryop.py __extend__ — SomeDict.method_* free functions

#[allow(dead_code)]
fn dict_method_get(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_dfl: &SomeValue,
) -> SomeValue {
    // unaryop.py method_get
    let position = ann.bookkeeper.current_position_key();
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key");
    s_self
        .dictdef
        .generalize_value(s_dfl)
        .expect("generalize_value");
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
fn dict_method_insert(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_value: &SomeValue,
) -> Result<SomeValue, AnnotatorError> {
    // Annotation of the old get(key) + setitem(key, value) pair:
    // get(key) defaults dfl to s_None (unaryop.py method_get), so the
    // result admits None (`ll_dict_insert` returns NULL on a miss);
    // setitem generalizes the stored value (binaryop.py setitem SomeDict).
    // `SomeInteger.noneify` raises `UnionError` (`model.py`), so a
    // non-nullable value type cannot represent the miss.
    let position = ann.bookkeeper.current_position_key();
    s_self
        .dictdef
        .generalize_key(s_key)
        .map_err(|e| AnnotatorError::new(e.to_string()))?;
    s_self.dictdef.generalize_value(&s_none()).map_err(|e| {
        AnnotatorError::new(format!(
            "dict.insert: value type cannot represent None (miss returns NULL): {e}"
        ))
    })?;
    s_self
        .dictdef
        .generalize_value(s_value)
        .map_err(|e| AnnotatorError::new(e.to_string()))?;
    Ok(s_self.dictdef.read_value(position))
}

#[allow(dead_code)]
fn dict_method_copy(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py method_copy — SomeDict(self.dictdef).
    SomeValue::Dict(super::model::SomeDict::new(s_self.dictdef.clone()))
}

#[allow(dead_code)]
fn dict_method_update(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_other: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:512-515
    //     if s_None.contains(dct2): return SomeImpossibleValue()
    //     dct1.dictdef.union(dct2.dictdef)
    // The explicit `return SomeImpossibleValue()` blocks; the union path
    // falls off the end (void op).
    if let SomeValue::None_(_) = s_other {
        return Some(s_impossible_value());
    }
    let other = match s_other {
        SomeValue::Dict(d) => d,
        _ => panic!("dict.update: other not SomeDict or SomeNone"),
    };
    s_self
        .dictdef
        .union_with(&other.dictdef)
        .expect("dictdef.union failed");
    None
}

#[allow(dead_code)]
fn dict_method_prepare_dict_update(
    _ann: &RPythonAnnotator,
    _s_self: &super::model::SomeDict,
    _s_num: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:517-518 — pass (void op).
    None
}

#[allow(dead_code)]
fn dict_method_keys(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py method_keys — bk.newlist(self.dictdef.read_key(bk.position_key)).
    let position = ann.bookkeeper.current_position_key();
    let s_key = s_self.dictdef.read_key(position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_key], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
fn dict_method_values(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py method_values
    let position = ann.bookkeeper.current_position_key();
    let s_value = s_self.dictdef.read_value(position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_value], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
fn dict_method_items(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py — bk.newlist(self.getanyitem(..., variant='items')).
    let position = ann.bookkeeper.current_position_key();
    let s_item = container_getanyitem(&SomeValue::Dict(s_self.clone()), Some("items"), position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
fn dict_method_iterkeys(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py method_iterkeys — SomeIterator(self, 'keys').
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["keys".into()],
    ))
}

#[allow(dead_code)]
fn dict_method_itervalues(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["values".into()],
    ))
}

#[allow(dead_code)]
fn dict_method_iteritems(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["items".into()],
    ))
}

#[allow(dead_code)]
fn dict_method_iterkeys_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["keys_with_hash".into()],
    ))
}

#[allow(dead_code)]
fn dict_method_iteritems_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["items_with_hash".into()],
    ))
}

#[allow(dead_code)]
fn dict_method_clear(
    _ann: &RPythonAnnotator,
    _s_self: &super::model::SomeDict,
) -> Option<SomeValue> {
    // unaryop.py:547-548 — pass (void op).
    None
}

#[allow(dead_code)]
fn dict_method_popitem(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py — return self.getanyitem(position, variant='items').
    let position = ann.bookkeeper.current_position_key();
    container_getanyitem(&SomeValue::Dict(s_self.clone()), Some("items"), position)
}

#[allow(dead_code)]
fn dict_method_pop(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_dfl: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py method_pop
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key");
    if let Some(s) = s_dfl {
        s_self
            .dictdef
            .generalize_value(s)
            .expect("generalize_value");
    }
    let position = ann.bookkeeper.current_position_key();
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
fn dict_method_contains_with_hash(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
) -> SomeValue {
    // unaryop.py method_contains_with_hash.
    let position = ann.bookkeeper.current_position_key();
    dict_contains(s_self, s_key, position)
}

#[allow(dead_code)]
fn dict_method_setitem_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
    s_value: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:566-567 — pair(self, s_key).setitem(s_value), falls off
    // the end (void op).
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key failed");
    s_self
        .dictdef
        .generalize_value(s_value)
        .expect("generalize_value failed");
    None
}

#[allow(dead_code)]
fn dict_method_getitem_with_hash(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
) -> SomeValue {
    // unaryop.py:570-574 — copy of binaryop.getitem_SomeDict.
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key failed");
    let position = ann.bookkeeper.current_position_key();
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
fn dict_method_delitem_with_hash(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    _s_hash: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:575-576 — pair(self, s_key).delitem(), falls off the end
    // (void op).
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key failed");
    None
}

#[allow(dead_code)]
fn dict_method_delitem_if_value_is(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_value: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py method_delitem_with_hash — setitem + delitem, falls off the end (void op).
    let _ = dict_method_setitem_with_hash(ann, s_self, s_key, &s_impossible_value(), s_value);
    dict_method_delitem_with_hash(ann, s_self, s_key, &s_impossible_value())
}

#[allow(dead_code)]
fn dict_method_move_to_end(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_last: &SomeValue,
) -> Option<SomeValue> {
    // unaryop.py:585-588 on SomeOrderedDict. Rust collapses SomeDict =
    // SomeOrderedDict (model.py), so the same analyzer lives on
    // SomeDict. assert + delitem, falls off the end (void op).
    assert!(
        s_bool().contains(s_last),
        "AnnotatorError: move_to_end(last) expects SomeBool"
    );
    dict_method_delitem_with_hash(ann, s_self, s_key, &s_impossible_value())
}

// unaryop.py:593-690 — @op.contains.register(SomeString|SomeUnicodeString)
// + class __extend__(SomeString, SomeUnicodeString) shared overrides

#[allow(non_snake_case)]
pub fn contains_String(ann: &RPythonAnnotator, hl: &HLOperation) -> SomeValue {
    let s_string = ann
        .annotation(&hl.args[0])
        .expect("contains(String): string unbound");
    let s_char = ann
        .annotation(&hl.args[1])
        .expect("contains(String): char unbound");
    if const_str_of(&s_char).as_deref() == Some("\0") {
        let mut r = SomeBool::new();
        let mut knowntypedata = std::collections::HashMap::new();
        let vars = match &hl.args[0] {
            Hlvalue::Variable(v) => vec![Rc::new(v.clone())],
            Hlvalue::Constant(_) => Vec::new(),
        };
        let s_nonnul = s_string
            .nonnulify()
            .expect("contains(String): nonnulify must succeed");
        add_knowntypedata(
            &mut knowntypedata,
            ExitCaseKey::Bool(false),
            &vars,
            s_nonnul,
        );
        r.set_knowntypedata(knowntypedata);
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

fn init_somestring_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — contains_String:
    //   if const '\0': knowntypedata-based refinement, else SomeObject fallback.
    //   can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::String,
        Specialization {
            apply: pure(contains_String),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::String,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("string.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — getslice: check_negative_slice + basestringclass(no_nul).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::String,
        Specialization {
            apply: pure(|ann, hl| {
                let s_self = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::String(s)) => s,
                    _ => panic!("string.getslice: arg 0 not SomeString"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("string.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("string.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::String(SomeString::new(false, s_self.inner.no_nul))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — len(self): immutablevalue(len(const)) if const, else SomeInteger(nonneg).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::String,
        Specialization {
            apply: pure(|ann, hl| {
                let s = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::String(s)) => s,
                    _ => panic!("string.len: arg 0 not SomeString"),
                };
                if s.is_constant()
                    && let Some(c) = &s.inner.base.const_box
                    && let ConstValue::ByteStr(v) = &c.value
                {
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(v.len() as i64)));
                    return SomeValue::Integer(i);
                }
                SomeValue::Integer(SomeInteger::new(true, false))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_someunicodestring_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:593-605 — contains on SomeUnicodeString (same path).
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: pure(contains_String),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — iter on SomeUnicodeString.
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("unicode.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — getslice on SomeUnicodeString (basestringclass=SomeUnicodeString).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: pure(|ann, hl| {
                let s_self = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::UnicodeString(s)) => s,
                    _ => panic!("unicode.getslice: arg 0 not SomeUnicodeString"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("unicode.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("unicode.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::UnicodeString(SomeUnicodeString::new(false, s_self.inner.no_nul))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_somebytearray_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — SomeByteArray.getslice: check + SomeByteArray().
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::ByteArray,
        Specialization {
            apply: pure(|ann, hl| {
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("bytearray.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("bytearray.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::ByteArray(super::model::SomeByteArray::new(false))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeChar, SomeUnicodeCodePoint) / SomeChar

fn init_somechar_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — len(self) = immutablevalue(1).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Char,
        Specialization {
            apply: pure(|_ann, _hl| {
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — ord(self) = SomeInteger(nonneg=True).
    register(
        reg,
        OpKind::Ord,
        SomeValueTag::Char,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_someunicodecp_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — len(self) = immutablevalue(1) on SomeUnicodeCodePoint.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::UnicodeCodePoint,
        Specialization {
            apply: pure(|_ann, _hl| {
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — ord(self): SomeInteger(nonneg=True).
    register(
        reg,
        OpKind::Ord,
        SomeValueTag::UnicodeCodePoint,
        Specialization {
            apply: pure(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py __extend__ / 720-748 / 776-797 — method_* free fns (String/Char)

/// RPython "basestringclass" picks SomeString vs SomeUnicodeString based
/// on the actual class of the SomeValue. SomeChar → SomeString,
/// SomeUnicodeCodePoint → SomeUnicodeString.
fn basestring_of(sv: &SomeValue, no_nul: bool) -> SomeValue {
    match sv {
        SomeValue::String(_) | SomeValue::Char(_) => {
            SomeValue::String(SomeString::new(false, no_nul))
        }
        SomeValue::UnicodeString(_) | SomeValue::UnicodeCodePoint(_) => {
            SomeValue::UnicodeString(SomeUnicodeString::new(false, no_nul))
        }
        other => panic!("basestring_of: unexpected {:?}", other),
    }
}

#[allow(dead_code)]
fn str_method_startswith(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_frag: &SomeValue,
) -> SomeValue {
    // unaryop.py method_startswith — const+const → immutablevalue(self.const.startswith(frag.const))
    //                       else s_Bool.
    if let (Some(a), Some(b)) = (const_str_of(s_self), const_str_of(s_frag)) {
        let mut r = SomeBool::new();
        r.base.const_box = Some(Constant::new(ConstValue::Bool(a.starts_with(&b))));
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
fn str_method_endswith(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_frag: &SomeValue,
) -> SomeValue {
    if let (Some(a), Some(b)) = (const_str_of(s_self), const_str_of(s_frag)) {
        let mut r = SomeBool::new();
        r.base.const_box = Some(Constant::new(ConstValue::Bool(a.ends_with(&b))));
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

fn const_str_of(sv: &SomeValue) -> Option<String> {
    let cb = match sv {
        SomeValue::String(s) => &s.inner.base.const_box,
        SomeValue::UnicodeString(s) => &s.inner.base.const_box,
        SomeValue::Char(s) => &s.inner.base.const_box,
        SomeValue::UnicodeCodePoint(s) => &s.inner.base.const_box,
        _ => return None,
    };
    cb.as_ref().and_then(|c| match &c.value {
        ConstValue::ByteStr(v) => String::from_utf8(v.clone()).ok(),
        ConstValue::UniStr(v) => Some(v.clone()),
        _ => None,
    })
}

fn const_int_of(sv: &SomeValue) -> Option<i64> {
    match sv.const_() {
        Some(ConstValue::Int(v)) => Some(*v),
        Some(ConstValue::UInt(v)) => Some(*v as i64),
        _ => None,
    }
}

fn stringish_no_nul(sv: &SomeValue) -> bool {
    match sv {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        SomeValue::Char(s) => s.inner.no_nul,
        SomeValue::UnicodeCodePoint(s) => s.inner.no_nul,
        _ => false,
    }
}

fn s_const_int(value: i64) -> SomeValue {
    let mut s = SomeInteger::new(value >= 0, false);
    s.base.const_box = Some(Constant::new(ConstValue::Int(value)));
    SomeValue::Integer(s)
}

#[allow(dead_code)]
fn str_method_find(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py — check_negative_slice(start,end,"find") → SomeInteger().
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "find");
    }
    SomeValue::Integer(SomeInteger::default())
}

#[allow(dead_code)]
fn str_method_rfind(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "rfind");
    }
    SomeValue::Integer(SomeInteger::default())
}

#[allow(dead_code)]
fn str_method_count(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "count");
    }
    SomeValue::Integer(SomeInteger::new(true, false))
}

fn str_method_strip_like(s_self: &SomeValue, s_chr: Option<&SomeValue>, opname: &str) -> SomeValue {
    // unaryop.py method_strip — strip/lstrip/rstrip share the same shape.
    if s_chr.is_none() && matches!(s_self, SomeValue::UnicodeString(_)) {
        panic!("AnnotatorError: unicode.{opname}() with no arg is not RPython");
    }
    basestring_of(s_self, stringish_no_nul(s_self))
}

#[allow(dead_code)]
fn str_method_strip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_chr: Option<&SomeValue>,
) -> SomeValue {
    str_method_strip_like(s_self, s_chr, "strip")
}

#[allow(dead_code)]
fn str_method_lstrip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_chr: Option<&SomeValue>,
) -> SomeValue {
    str_method_strip_like(s_self, s_chr, "lstrip")
}

#[allow(dead_code)]
fn str_method_rstrip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_chr: Option<&SomeValue>,
) -> SomeValue {
    str_method_strip_like(s_self, s_chr, "rstrip")
}

#[allow(dead_code)]
fn str_method_join(ann: &RPythonAnnotator, s_self: &SomeValue, s_list: &SomeValue) -> SomeValue {
    // unaryop.py method_join.
    if s_none().contains(s_list) {
        return SomeValue::Impossible;
    }
    let SomeValue::List(s_list) = s_list else {
        panic!("AnnotatorError: string.join() argument must be a list");
    };
    let position = ann.bookkeeper.current_position_key();
    let s_item = s_list.listdef.read_item(position);
    if s_none().contains(&s_item) {
        // upstream unaryop.py:653-656:
        //     if isinstance(self, SomeUnicodeString):
        //         return immutablevalue(u"")
        //     return immutablevalue("")
        // The SomeString branch routes through bookkeeper.immutablevalue
        // so no_nul / const_box carry the same invariants as any other
        // empty-string constant.
        let s_empty = ann
            .bookkeeper
            .immutablevalue(&ConstValue::byte_str(""))
            .expect("immutablevalue(\"\") must succeed");
        return match s_self {
            SomeValue::UnicodeString(_) => {
                let SomeValue::String(s) = s_empty else {
                    unreachable!("immutablevalue(\"\") always returns SomeString");
                };
                let mut u = SomeUnicodeString::new(false, s.inner.no_nul);
                u.inner.base.const_box = s.inner.base.const_box;
                SomeValue::UnicodeString(u)
            }
            _ => s_empty,
        };
    }
    let no_nul = stringish_no_nul(s_self) && stringish_no_nul(&s_item);
    basestring_of(s_self, no_nul)
}

#[allow(dead_code)]
fn str_method_split(
    ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_patt: &SomeValue,
    s_max: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py method_split.
    let max_value = s_max.and_then(const_int_of).unwrap_or(-1);
    let no_nul = if max_value == -1
        && const_str_of(s_patt)
            .map(|p| p.chars().count() == 1 && p.starts_with('\0'))
            .unwrap_or(false)
    {
        true
    } else {
        stringish_no_nul(s_self)
    };
    let s_item = basestring_of(s_self, no_nul);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
fn str_method_rsplit(
    ann: &RPythonAnnotator,
    s_self: &SomeValue,
    _s_patt: &SomeValue,
    _s_max: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py method_rsplit.
    let s_item = basestring_of(s_self, stringish_no_nul(s_self));
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
fn str_method_upper(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    // unaryop.py method_upper / 793-797 — SomeChar.upper returns self;
    // SomeString/SomeUnicodeString inherit basestringclass().
    match _s_self {
        SomeValue::Char(_) => _s_self.clone(),
        _ => basestring_of(_s_self, false),
    }
}

#[allow(dead_code)]
fn str_method_lower(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    match _s_self {
        SomeValue::Char(_) => _s_self.clone(),
        _ => basestring_of(_s_self, false),
    }
}

#[allow(dead_code)]
fn str_method_isdigit(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
fn str_method_isalpha(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
fn descr_isalnum(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
fn str_method_replace(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    _s1: &SomeValue,
    s2: &SomeValue,
) -> SomeValue {
    // unaryop.py method_replace — basestringclass(no_nul=self.no_nul and s2.no_nul).
    let self_no_nul = match s_self {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        _ => false,
    };
    let s2_no_nul = match s2 {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        SomeValue::Char(s) => s.inner.no_nul,
        SomeValue::UnicodeCodePoint(s) => s.inner.no_nul,
        _ => false,
    };
    basestring_of(s_self, self_no_nul && s2_no_nul)
}

#[allow(dead_code)]
fn str_method_splitlines(ann: &RPythonAnnotator, s_self: &SomeValue) -> SomeValue {
    // unaryop.py:744-748.
    let s_item = basestring_of(s_self, false);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_item], None)
        .expect("bookkeeper.newlist failed");
    s_list.listdef.resize().expect("listdef.resize failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
fn descr_format(_ann: &RPythonAnnotator, _s_self: &SomeValue, _args: &[SomeValue]) -> SomeValue {
    // unaryop.py method_format — always raises.
    panic!("AnnotatorError: Method format() is not RPython")
}

#[allow(dead_code)]
fn char_method_isspace(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
fn char_method_islower(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
fn char_method_isupper(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

/// `StringBuilder.append(self, s_str)` (rstring.py) — void; the
/// model asserts the arg is str/char and the call binds Impossible.
fn stringbuilder_method_append(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeStringBuilder,
    s_str: &SomeValue,
) -> Option<SomeValue> {
    let _ = s_self.method_append(s_str);
    None
}

/// `StringBuilder.append_slice(self, s_str, s_start, s_end)` (rstring.py) —
/// void; the model asserts a str arg with integer bounds and the call binds
/// Impossible.
fn stringbuilder_method_append_slice(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeStringBuilder,
    s_str: &SomeValue,
    s_start: &SomeValue,
    s_end: &SomeValue,
) -> Option<SomeValue> {
    let _ = s_self.method_append_slice(s_str, s_start, s_end);
    None
}

/// `StringBuilder.build(self)` (rstring.py) → `SomeString`.
fn stringbuilder_method_build(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeStringBuilder,
) -> SomeValue {
    s_self.method_build()
}

/// `StringBuilder.getlength(self)` (rstring.py) → `SomeInteger`.
fn stringbuilder_method_getlength(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeStringBuilder,
) -> SomeValue {
    s_self.method_getlength()
}

/// `UnicodeBuilder.getlength(self)` (rstring.py) → `SomeInteger`.
fn unicodebuilder_method_getlength(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeUnicodeBuilder,
) -> SomeValue {
    s_self.method_getlength()
}

/// RPython `SomeObject.find_method(self, name)` (unaryop.py).
pub(crate) fn find_method(s_self: &SomeValue, name: &str) -> Option<SomeBuiltinMethod> {
    let analyser_name = match s_self {
        SomeValue::List(_) => match name {
            "append" => "list_method_append",
            "extend" => "list_method_extend",
            "reverse" => "descr_reverse",
            "insert" => "descr_insert",
            "remove" => "descr_remove",
            "pop" => "list_method_pop",
            "index" => "list_method_index",
            // Raw-pointer `p.is_null()` on a `SomeList` receiver (the list
            // backing carries a `Ptr` concretetype); the getattr annotator
            // answers the same probe with `ptr_method_is_null`, so mirror it
            // here for the rtyper's method-existence check.
            "is_null" => "ptr_method_is_null",
            _ => return None,
        },
        SomeValue::Dict(_) => match name {
            "get" | "setdefault" => "dict_method_get",
            "insert" => "dict_method_insert",
            "copy" => "dict_method_copy",
            "update" => "dict_method_update",
            "__prepare_dict_update" => "dict_method_prepare_dict_update",
            "keys" => "dict_method_keys",
            "values" => "dict_method_values",
            "items" => "dict_method_items",
            "iterkeys" => "dict_method_iterkeys",
            "itervalues" => "dict_method_itervalues",
            "iteritems" => "dict_method_iteritems",
            "iterkeys_with_hash" => "dict_method_iterkeys_with_hash",
            "iteritems_with_hash" => "dict_method_iteritems_with_hash",
            "clear" => "dict_method_clear",
            "popitem" => "dict_method_popitem",
            "pop" => "dict_method_pop",
            "contains_with_hash" => "dict_method_contains_with_hash",
            "setitem_with_hash" => "dict_method_setitem_with_hash",
            "getitem_with_hash" => "dict_method_getitem_with_hash",
            "delitem_with_hash" => "dict_method_delitem_with_hash",
            "delitem_if_value_is" => "dict_method_delitem_if_value_is",
            "move_to_end" => "dict_method_move_to_end",
            // A raw-pointer `p.is_null()` whose receiver Charon erases to
            // a `SomeDict` (the dict-typed field carries a `Ptr(dicttable)`
            // concretetype).  The getattr annotator answers the same probe
            // with `ptr_method_is_null`; mirror it here so the rtyper's
            // `rtype_getattr` method-existence check finds it and forwards
            // the receiver to `OrderedDictRepr::rtype_method("is_null")`.
            "is_null" => "ptr_method_is_null",
            _ => return None,
        },
        SomeValue::String(_) => match name {
            "startswith" => "str_method_startswith",
            "endswith" => "str_method_endswith",
            "find" => "str_method_find",
            "rfind" => "str_method_rfind",
            "count" => "str_method_count",
            "strip" => "str_method_strip",
            "lstrip" => "str_method_lstrip",
            "rstrip" => "str_method_rstrip",
            "join" => "str_method_join",
            "split" => "str_method_split",
            "rsplit" => "str_method_rsplit",
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "descr_isalnum",
            "replace" => "str_method_replace",
            "splitlines" => "str_method_splitlines",
            "format" => "descr_format",
            _ => return None,
        },
        SomeValue::UnicodeString(_) => match name {
            "startswith" => "str_method_startswith",
            "endswith" => "str_method_endswith",
            "find" => "str_method_find",
            "rfind" => "str_method_rfind",
            "count" => "str_method_count",
            "strip" => "str_method_strip",
            "lstrip" => "str_method_lstrip",
            "rstrip" => "str_method_rstrip",
            "join" => "str_method_join",
            "split" => "str_method_split",
            "rsplit" => "str_method_rsplit",
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "descr_isalnum",
            "replace" => "str_method_replace",
            "format" => "descr_format",
            _ => return None,
        },
        SomeValue::Char(_) => match name {
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "descr_isalnum",
            "isspace" => "char_method_isspace",
            "islower" => "char_method_islower",
            "isupper" => "char_method_isupper",
            _ => return None,
        },
        SomeValue::UnicodeCodePoint(_) => match name {
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "descr_isalnum",
            _ => return None,
        },
        // rstring.py — the `StringBuilder` call surface.  Only the
        // methods the rtyper lowers (`StringBuilderRepr::rtype_method`) are
        // exposed here: append/append_slice/build/getlength.
        SomeValue::StringBuilder(_) => match name {
            "append" => "stringbuilder_method_append",
            "append_slice" => "stringbuilder_method_append_slice",
            "build" => "stringbuilder_method_build",
            "getlength" => "stringbuilder_method_getlength",
            _ => return None,
        },
        // The Rust `Vec` methods `RustVecRepr::rtype_method` lowers.
        SomeValue::RustVec(_) => match name {
            "append" => "rustvec_method_append",
            "reverse" => "rustvec_method_reverse",
            "free" => "rustvec_method_free",
            "items" => "rustvec_method_items",
            "extend_from_slice" => "rustvec_method_extend_from_slice",
            "extend" => "rustvec_method_extend",
            _ => return None,
        },
        // rstring.py — the `UnicodeBuilder` surface the rtyper
        // lowers so far (`UnicodeBuilderRepr::rtype_method`): getlength.
        SomeValue::UnicodeBuilder(_) => match name {
            "getlength" => "unicodebuilder_method_getlength",
            _ => return None,
        },
        // unaryop.py — `method_next = next` on SomeIterator.  Lets
        // `getattr(iter, "next")` (emitted by `next_SomeInstance` when an
        // `OpKind::Next` lowers on an instance-shaped receiver that later
        // refines to an iterator) resolve to a bound builtin method instead
        // of the generic "Cannot find attribute" dead-end.
        SomeValue::Iterator(_) => match name {
            "next" => "iterator_method_next",
            _ => return None,
        },
        _ => return None,
    };
    Some(SomeBuiltinMethod::new(analyser_name, s_self.clone(), name))
}

fn builtin_method_receiver_error(method: &SomeBuiltinMethod) -> AnnotatorError {
    AnnotatorError::new(format!(
        "SomeBuiltinMethod.call(): receiver/type mismatch for {}",
        method.methodname
    ))
}

fn builtin_method_arg_error(method: &SomeBuiltinMethod) -> AnnotatorError {
    AnnotatorError::new(format!(
        "SomeBuiltinMethod.call(): wrong argument shape for {}",
        method.methodname
    ))
}

/// Model upstream `SomeBuiltinMethod.call(self, args)` argument
/// binding (unaryop.py):
///
/// ```python
/// def call(self, args, implicit_init=False):
///     args_s, kwds = args.unpack()
///     # prefix keyword arguments with 's_'
///     kwds_s = {}
///     for key, s_value in kwds.items():
///         kwds_s['s_'+key] = s_value
///     return self.analyser(self.s_self, *args_s, **kwds_s)
/// ```
///
/// In RPython the final `self.analyser(self.s_self, *args_s, **kwds_s)`
/// relies on Python's native calling convention to pair positional
/// args + `s_`-prefixed kwargs against the `method_X(self, ...)`
/// signature, plus the method's default values for missing positions.
/// Rust has no reflection, so the port binds the arguments explicitly
/// through [`ArgumentsForTranslation::match_signature`] (the Rust
/// port of `argument.py:126-133`) with the analyser's formal
/// `argnames` and `defaults` threaded in from each dispatch arm.
/// The resulting `Vec<SomeValue>` matches positional order 1:1 with
/// the analyser's formals, letting the call sites destructure via
/// `let [..] = scope.as_slice()` without inspecting `kwds` again.
fn bind_builtin_method_args(
    args_s: Vec<Option<SomeValue>>,
    kwds: std::collections::HashMap<String, Option<SomeValue>>,
    argnames: &[&str],
    defaults: Option<Vec<SomeValue>>,
    analyser_name: &str,
) -> Result<Vec<SomeValue>, AnnotatorError> {
    let signature = crate::flowspace::argument::Signature::new(
        argnames.iter().map(|name| (*name).to_string()).collect(),
        None,
        None,
    );
    // upstream: `kwds_s['s_'+key] = s_value` — `s_value` may be None
    // (unbound caller arg).  The Option-aware threading propagates
    // through `match_signature`'s `scope_w` carrier, which itself is
    // `Vec<Option<SomeValue>>` so unbound slots reach the per-touch
    // unwrap below.  Pre-prefixed dict stays `Option<SomeValue>`-valued
    // to preserve None across the analyser boundary.
    let prefixed_kwds: std::collections::HashMap<String, Option<SomeValue>> = kwds
        .into_iter()
        .map(|(key, value)| (format!("s_{key}"), value))
        .collect();
    // upstream `unaryop.py call` passes `args_s` straight through —
    // an unbound caller cell would AttributeError on first attribute
    // touch in the analyser body.  Thread `Vec<Option<SomeValue>>`
    // through `match_signature` (the binding layer is Option-aware),
    // then panic per-slot at the boundary so the failure surfaces with
    // the analyser name + slot index.
    let call_args =
        super::argument::ArgumentsForTranslation::new(args_s, Some(prefixed_kwds), None);
    let scope_w = call_args
        .match_signature(&signature, defaults.as_deref())
        .map_err(|err| {
            AnnotatorError::new(format!("SomeBuiltinMethod.call(): {}", err.getmsg()))
        })?;
    Ok(scope_w
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            c.unwrap_or_else(|| {
                panic!(
                    "{analyser_name}: positional arg {i} is unannotated \
                     (upstream `args_s[{i}].knowntype` AttributeError)",
                )
            })
        })
        .collect())
}

/// RPython `SomeBuiltinMethod.call(self, args, implicit_init=False)`
/// dispatch helper (unaryop.py).
pub(crate) fn call_builtin_method(
    ann: &RPythonAnnotator,
    method: &SomeBuiltinMethod,
    args: &super::argument::ArgumentsForTranslation,
) -> Result<Option<SomeValue>, AnnotatorError> {
    let (args_s, kwds) = args
        .unpack()
        .map_err(|err| AnnotatorError::new(err.getmsg()))?;
    // `unaryop.py SomeBuiltinMethod.call` passes args_s
    // through `bookkeeper.consider_call_site` and into the analyser
    // body.  Optional-default methods (list.pop(s_index=None),
    // dict.pop(s_dfl=None), str.find(start=None, end=None)) accept
    // missing positional slots that bind to Python None via
    // signature defaults.  Thread `Vec<Option<SomeValue>>` through
    // bind_builtin_method_args so the per-slot panic only fires on
    // post-default-binding `None`, matching upstream's first
    // attribute-touch AttributeError surface.

    let result = match method.analyser_name.as_str() {
        "list_method_append" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_value"], None, &method.analyser_name)?;
            let [s_value] = scope.as_slice() else {
                unreachable!();
            };
            // Void analyser (None = no result): early-return the threaded
            // Option so the simple_call binds Impossible without blocking.
            return Ok(list_method_append(ann, s_self, s_value));
        }
        "list_method_extend" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_iterable"],
                None,
                &method.analyser_name,
            )?;
            let [s_iterable] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(list_method_extend(ann, s_self, s_iterable));
        }
        "descr_reverse" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(descr_reverse(ann, s_self));
        }
        "descr_insert" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_index", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_index, s_value] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(descr_insert(ann, s_self, s_index, s_value));
        }
        "descr_remove" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_value"], None, &method.analyser_name)?;
            let [s_value] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(descr_remove(ann, s_self, s_value));
        }
        "list_method_pop" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_index"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_index] = scope.as_slice() else {
                unreachable!();
            };
            if matches!(s_index, SomeValue::None_(_)) {
                list_method_pop(ann, s_self, None)
            } else {
                list_method_pop(ann, s_self, Some(s_index))
            }
        }
        "list_method_index" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_value"], None, &method.analyser_name)?;
            let [s_value] = scope.as_slice() else {
                unreachable!();
            };
            list_method_index(ann, s_self, s_value)
        }
        "dict_method_get" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_dfl"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_key, s_dfl] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_get(ann, s_self, s_key, s_dfl)
        }
        "dict_method_insert" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_value] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_insert(ann, s_self, s_key, s_value)?
        }
        "dict_method_copy" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_copy(ann, s_self)
        }
        "dict_method_update" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_other"], None, &method.analyser_name)?;
            let [s_other] = scope.as_slice() else {
                unreachable!();
            };
            // Void on the union path, blocks (Some(Impossible)) on the
            // s_None receiver path — see dict_method_update.
            return Ok(dict_method_update(ann, s_self, s_other));
        }
        "dict_method_prepare_dict_update" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_num"], None, &method.analyser_name)?;
            let [s_num] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(dict_method_prepare_dict_update(ann, s_self, s_num));
        }
        "dict_method_keys" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_keys(ann, s_self)
        }
        "dict_method_values" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_values(ann, s_self)
        }
        "dict_method_items" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_items(ann, s_self)
        }
        "dict_method_iterkeys" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iterkeys(ann, s_self)
        }
        "dict_method_itervalues" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_itervalues(ann, s_self)
        }
        "dict_method_iteritems" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iteritems(ann, s_self)
        }
        "dict_method_iterkeys_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iterkeys_with_hash(ann, s_self)
        }
        "dict_method_iteritems_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_iteritems_with_hash(ann, s_self)
        }
        "dict_method_clear" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(dict_method_clear(ann, s_self));
        }
        "dict_method_popitem" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_popitem(ann, s_self)
        }
        "dict_method_pop" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_dfl"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_key, s_dfl] = scope.as_slice() else {
                unreachable!();
            };
            if matches!(s_dfl, SomeValue::None_(_)) {
                dict_method_pop(ann, s_self, s_key, None)
            } else {
                dict_method_pop(ann, s_self, s_key, Some(s_dfl))
            }
        }
        "dict_method_contains_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_contains_with_hash(ann, s_self, s_key, s_hash)
        }
        "dict_method_setitem_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash, s_value] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(dict_method_setitem_with_hash(
                ann, s_self, s_key, s_hash, s_value,
            ));
        }
        "dict_method_getitem_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash] = scope.as_slice() else {
                unreachable!();
            };
            dict_method_getitem_with_hash(ann, s_self, s_key, s_hash)
        }
        "dict_method_delitem_with_hash" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_hash"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_hash] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(dict_method_delitem_with_hash(ann, s_self, s_key, s_hash));
        }
        "dict_method_delitem_if_value_is" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_value"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_value] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(dict_method_delitem_if_value_is(ann, s_self, s_key, s_value));
        }
        "dict_method_move_to_end" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_key", "s_last"],
                None,
                &method.analyser_name,
            )?;
            let [s_key, s_last] = scope.as_slice() else {
                unreachable!();
            };
            return Ok(dict_method_move_to_end(ann, s_self, s_key, s_last));
        }
        "str_method_startswith" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_frag"], None, &method.analyser_name)?;
            let [s_frag] = scope.as_slice() else {
                unreachable!();
            };
            str_method_startswith(ann, &method.s_self, s_frag)
        }
        "str_method_endswith" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_frag"], None, &method.analyser_name)?;
            let [s_frag] = scope.as_slice() else {
                unreachable!();
            };
            str_method_endswith(ann, &method.s_self, s_frag)
        }
        "str_method_find" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_frag", "s_start", "s_end"],
                Some(vec![s_none(), s_none()]),
                &method.analyser_name,
            )?;
            let [s_frag, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            str_method_find(
                ann,
                &method.s_self,
                s_frag,
                if matches!(s_start, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_start)
                },
                if matches!(s_end, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_end)
                },
            )
        }
        "str_method_rfind" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_frag", "s_start", "s_end"],
                Some(vec![s_none(), s_none()]),
                &method.analyser_name,
            )?;
            let [s_frag, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            str_method_rfind(
                ann,
                &method.s_self,
                s_frag,
                if matches!(s_start, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_start)
                },
                if matches!(s_end, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_end)
                },
            )
        }
        "str_method_count" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_frag", "s_start", "s_end"],
                Some(vec![s_none(), s_none()]),
                &method.analyser_name,
            )?;
            let [s_frag, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            str_method_count(
                ann,
                &method.s_self,
                s_frag,
                if matches!(s_start, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_start)
                },
                if matches!(s_end, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_end)
                },
            )
        }
        "str_method_strip" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_chr"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_chr] = scope.as_slice() else {
                unreachable!();
            };
            str_method_strip(
                ann,
                &method.s_self,
                if matches!(s_chr, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_chr)
                },
            )
        }
        "str_method_lstrip" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_chr"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_chr] = scope.as_slice() else {
                unreachable!();
            };
            str_method_lstrip(
                ann,
                &method.s_self,
                if matches!(s_chr, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_chr)
                },
            )
        }
        "str_method_rstrip" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_chr"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [s_chr] = scope.as_slice() else {
                unreachable!();
            };
            str_method_rstrip(
                ann,
                &method.s_self,
                if matches!(s_chr, SomeValue::None_(_)) {
                    None
                } else {
                    Some(s_chr)
                },
            )
        }
        "str_method_join" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_list"], None, &method.analyser_name)?;
            let [s_list] = scope.as_slice() else {
                unreachable!();
            };
            str_method_join(ann, &method.s_self, s_list)
        }
        "str_method_split" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_patt", "s_max"],
                Some(vec![s_const_int(-1)]),
                &method.analyser_name,
            )?;
            let [s_patt, s_max] = scope.as_slice() else {
                unreachable!();
            };
            str_method_split(ann, &method.s_self, s_patt, Some(s_max))
        }
        "str_method_rsplit" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_patt", "s_max"],
                Some(vec![s_const_int(-1)]),
                &method.analyser_name,
            )?;
            let [s_patt, s_max] = scope.as_slice() else {
                unreachable!();
            };
            str_method_rsplit(ann, &method.s_self, s_patt, Some(s_max))
        }
        "str_method_upper" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_upper(ann, &method.s_self)
        }
        "str_method_lower" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_lower(ann, &method.s_self)
        }
        "str_method_isdigit" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_isdigit(ann, &method.s_self)
        }
        "str_method_isalpha" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            str_method_isalpha(ann, &method.s_self)
        }
        "descr_isalnum" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            descr_isalnum(ann, &method.s_self)
        }
        "str_method_replace" => {
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s1", "s2"], None, &method.analyser_name)?;
            let [s1, s2] = scope.as_slice() else {
                unreachable!();
            };
            str_method_replace(ann, &method.s_self, s1, s2)
        }
        "str_method_splitlines" => {
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_keep_newlines"],
                Some(vec![s_none()]),
                &method.analyser_name,
            )?;
            let [_s_keep_newlines] = scope.as_slice() else {
                unreachable!();
            };
            str_method_splitlines(ann, &method.s_self)
        }
        "descr_format" => {
            if !kwds.is_empty() {
                return Err(builtin_method_arg_error(method));
            }
            // upstream `descr_format` always raises — the args are
            // unused, so the Option layer can be flattened with
            // Impossible at this single call site.
            let args_concrete: Vec<SomeValue> = args_s
                .iter()
                .map(|s| s.clone().unwrap_or_else(super::model::s_impossible_value))
                .collect();
            descr_format(ann, &method.s_self, &args_concrete)
        }
        "char_method_isspace" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            char_method_isspace(ann, &method.s_self)
        }
        "char_method_islower" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            char_method_islower(ann, &method.s_self)
        }
        "char_method_isupper" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            char_method_isupper(ann, &method.s_self)
        }
        // Raw-pointer `p.is_null()` on a classdef-less `SomeInstance`
        // (pointer-erasure receiver; bound in the SomeInstance.getattr
        // `is_null` arm).  Zero-arg; the call always yields `SomeBool`,
        // mirroring lltype `_ptr.is_null`.
        "ptr_method_is_null" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            super::model::s_bool()
        }
        // `Option<RawStruct>::as_ref` / `<*mut T>::as_ref` on a SomePtr
        // receiver: the pointer word is the option. Zero-arg identity.
        "ptr_method_as_ref" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            method.s_self.as_ref().clone()
        }
        // `PyError::to_exc_object(&self) -> PyObjectRef` — the boxed
        // `W_BaseException`.  Zero positional args; the result is a
        // `PyObjectRef` instance (bound in the SomeInstance.getattr
        // `to_exc_object` arm).  The codewriter keeps the real conversion
        // residual (`for_impl_method("PyError", "to_exc_object")`).
        "pyerror_method_to_exc_object" => {
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            ann.bookkeeper.project_struct_field_type("PyObjectRef")
        }
        // unaryop.py — `next(self)` / `method_next = next`.  The
        // bound iterator is the receiver; next takes no positional args and
        // returns the element type via the shared `someiterator_next` body
        // (same analyser as the `OpKind::Next` specialization).
        "iterator_method_next" => {
            let SomeValue::Iterator(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            someiterator_next(ann, s_self)
        }
        // The item annotation is fixed by the Rust item type, so the
        // methods neither mutate nor generalize it; all three are void.
        "rustvec_method_append"
        | "rustvec_method_reverse"
        | "rustvec_method_free"
        | "rustvec_method_extend_from_slice"
        | "rustvec_method_extend" => {
            let SomeValue::RustVec(_) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let params: &[&str] = match method.analyser_name.as_str() {
                "rustvec_method_append" => &["s_item"],
                // The `(items, length)` words of the source slice.
                "rustvec_method_extend_from_slice" => &["s_items", "s_length"],
                // `rlist.ll_extend(l1, l2)`: `l2` is the GcArray word.
                "rustvec_method_extend" => &["s_l2"],
                _ => &[],
            };
            bind_builtin_method_args(args_s, kwds, params, None, &method.analyser_name)?;
            return Ok(None);
        }
        // `ll_items`: the item pointer, a raw address word (`usize`).
        "rustvec_method_items" => {
            let SomeValue::RustVec(_) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            return Ok(Some(SomeValue::Integer(SomeInteger::new(true, true))));
        }
        "stringbuilder_method_append" => {
            let SomeValue::StringBuilder(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope =
                bind_builtin_method_args(args_s, kwds, &["s_str"], None, &method.analyser_name)?;
            let [s_str] = scope.as_slice() else {
                unreachable!();
            };
            // Void analyser (None = no result): early-return the threaded
            // Option so the simple_call binds Impossible without blocking.
            return Ok(stringbuilder_method_append(ann, s_self, s_str));
        }
        "stringbuilder_method_append_slice" => {
            let SomeValue::StringBuilder(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(
                args_s,
                kwds,
                &["s_str", "s_start", "s_end"],
                None,
                &method.analyser_name,
            )?;
            let [s_str, s_start, s_end] = scope.as_slice() else {
                unreachable!();
            };
            // Void analyser: early-return the threaded Option (see append).
            return Ok(stringbuilder_method_append_slice(
                ann, s_self, s_str, s_start, s_end,
            ));
        }
        "stringbuilder_method_build" => {
            let SomeValue::StringBuilder(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            stringbuilder_method_build(ann, s_self)
        }
        "stringbuilder_method_getlength" => {
            let SomeValue::StringBuilder(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            stringbuilder_method_getlength(ann, s_self)
        }
        "unicodebuilder_method_getlength" => {
            let SomeValue::UnicodeBuilder(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let scope = bind_builtin_method_args(args_s, kwds, &[], None, &method.analyser_name)?;
            let [] = scope.as_slice() else {
                unreachable!();
            };
            unicodebuilder_method_getlength(ann, s_self)
        }
        _ => {
            return Err(AnnotatorError::new(format!(
                "SomeBuiltinMethod.call(): unknown analyser {}",
                method.analyser_name
            )));
        }
    };
    // Value analysers return a concrete annotation (possibly Bottom =
    // SomeImpossibleValue when reading an empty container, which blocks);
    // the void analysers above already early-returned their `None`.
    Ok(Some(result))
}

// unaryop.py — class __extend__(SomeIterator)

fn init_someiterator_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — iter(self): return self. can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Iterator,
        Specialization {
            apply: pure(|ann, hl| ann.annotation(&hl.args[0]).expect("iterator.iter: unbound")),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — next(self):
    //   position = getbookkeeper().position_key
    //   if s_None.contains(self.s_container): return s_ImpossibleValue
    //   if self.variant and self.variant[0] == "enumerate":
    //       s_item = self.s_container.getanyitem(position)
    //       return SomeTuple((SomeInteger(nonneg=True), s_item))
    //   variant = self.variant
    //   if variant == ("reversed",): variant = ()
    //   return self.s_container.getanyitem(position, *variant)
    register(
        reg,
        OpKind::Next,
        SomeValueTag::Iterator,
        Specialization {
            apply: pure(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("iterator.next: unbound");
                let it = match s {
                    SomeValue::Iterator(i) => i,
                    _ => panic!("iterator.next: arg 0 not SomeIterator"),
                };
                someiterator_next(ann, &it)
            }),
            can_only_throw: CanOnlyThrow::Callable(Box::new(|args_s| {
                // upstream `_can_only_throw` (unaryop.py):
                //   can_throw = [StopIteration]
                //   if isinstance(self.s_container, SomeDict):
                //       can_throw.append(RuntimeError)
                //   return can_throw
                let mut throws = vec![BuiltinException::StopIteration];
                if let Some(SomeValue::Iterator(i)) = args_s.first()
                    && matches!(&*i.s_container, SomeValue::Dict(_))
                {
                    throws.push(BuiltinException::RuntimeError);
                }
                Some(throws)
            })),
        },
    );
}

/// RPython `SomeIterator.next(self)` (unaryop.py). Extracted
/// into a free function so `list_method_extend` (unaryop.py)
/// can call `s_iterable.iter().next()` directly without going through
/// the HLOperation dispatch framework — upstream also calls it as a
/// plain method invocation on a SomeIterator value.
fn someiterator_next(ann: &RPythonAnnotator, it: &SomeIterator) -> SomeValue {
    let position = ann.bookkeeper.current_position_key();
    // upstream: `if s_None.contains(self.s_container): return s_ImpossibleValue`.
    if matches!(&*it.s_container, SomeValue::None_(_)) {
        return s_impossible_value();
    }
    // upstream: `if self.variant and self.variant[0] == "enumerate": ...`.
    if it.variant.first().map(String::as_str) == Some("enumerate") {
        let s_item = container_getanyitem(&it.s_container, None, position.clone());
        return SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Integer(SomeInteger::new(true, false)),
            s_item,
        ]));
    }
    // upstream: `if variant == ("reversed",): variant = ()`.
    let variant: Option<&str> = if it.variant.len() == 1 && it.variant[0] == "reversed" {
        None
    } else {
        it.variant.first().map(String::as_str)
    };
    container_getanyitem(&it.s_container, variant, position)
}

/// RPython `container.getanyitem(position, *variant)` dispatch —
/// pulled out into a free function so [`SomeIterator.next`] can call
/// it once for the enumerate branch (with variant forced to `()`) and
/// once for the default branch (variant = `it.variant`).
///
/// Upstream definitions:
/// * SomeTuple (unaryop.py): `unionof(*self.items)`.
/// * SomeList (unaryop.py):  `self.listdef.read_item(position)`.
/// * SomeDict (unaryop.py):  variant-dispatched.
/// * SomeStringOrUnicode (unaryop.py): `self.basecharclass()`.
pub(crate) fn container_getanyitem(
    s_container: &SomeValue,
    variant: Option<&str>,
    position: Option<super::bookkeeper::PositionKey>,
) -> SomeValue {
    use super::super::flowspace::argument::Signature;
    let _ = Signature::new; // silence dead-imports on cold paths
    match s_container {
        SomeValue::Tuple(t) => {
            if t.items.is_empty() {
                s_impossible_value()
            } else {
                super::model::unionof(t.items.iter()).expect("getanyitem(tuple): unionof failed")
            }
        }
        SomeValue::List(l) => l.listdef.read_item(position),
        SomeValue::RustVec(v) => (*v.s_item).clone(),
        SomeValue::Dict(d) => {
            // unaryop.py getanyitem — per-variant dispatch.
            match variant.unwrap_or("keys") {
                "keys" => d.dictdef.read_key(position),
                "values" => d.dictdef.read_value(position),
                "items" => {
                    let s_key = d.dictdef.read_key(position.clone());
                    let s_value = d.dictdef.read_value(position);
                    if matches!(s_key, SomeValue::Impossible)
                        || matches!(s_value, SomeValue::Impossible)
                    {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![s_key, s_value]))
                    }
                }
                "items_with_hash" => {
                    let s_key = d.dictdef.read_key(position.clone());
                    let s_value = d.dictdef.read_value(position);
                    if matches!(s_key, SomeValue::Impossible)
                        || matches!(s_value, SomeValue::Impossible)
                    {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![
                            s_key,
                            s_value,
                            SomeValue::Integer(SomeInteger::default()),
                        ]))
                    }
                }
                "keys_with_hash" => {
                    let s_key = d.dictdef.read_key(position);
                    if matches!(s_key, SomeValue::Impossible) {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![
                            s_key,
                            SomeValue::Integer(SomeInteger::default()),
                        ]))
                    }
                }
                other => panic!("getanyitem(dict): unknown variant {other:?}"),
            }
        }
        // unaryop.py — `SomeStringOrUnicode.getanyitem` returns
        // `self.basecharclass()`. model.py:326-329 assigns
        //   SomeString.basecharclass = SomeChar
        //   SomeUnicodeString.basecharclass = SomeUnicodeCodePoint
        // and the class is called with no arguments — so the char's
        // `no_nul` defaults to False regardless of the string's.
        SomeValue::String(_) => SomeValue::Char(super::model::SomeChar::new(false)),
        SomeValue::UnicodeString(_) => {
            SomeValue::UnicodeCodePoint(super::model::SomeUnicodeCodePoint::new(false))
        }
        // A Rust fixed array is a `SomeInstance` of its shaped `Array<T;N>`
        // class, not a `SomeList`. RPython's fixed-size list is still a
        // `SomeList` whose `SomeIterator.next` reads `listdef.read_item`
        // (`ListItem.read_locations`). The item annotation here is the
        // constructor's `__pos_N` cells; `record_getattr` is the reader set
        // `ClassDef.generalize_attr` reflows (`Attribute.read_locations`).
        SomeValue::Instance(inst) => shaped_array_getanyitem(inst),
        other => panic!("getanyitem: unsupported container {other:?}"),
    }
}

fn instance_is_shaped_array(s: &SomeValue) -> bool {
    let SomeValue::Instance(inst) = s else {
        return false;
    };
    inst.classdef
        .as_ref()
        .is_some_and(|cd| majit_ir::descr::is_shaped_array_name(&cd.borrow().name))
}

/// Item annotation of a shaped `Array<T;N>` instance: the union of the
/// constructor-written `__pos_N` cells. Registers each field as a getattr
/// reader at the current bookkeeper position so a later `generalize_attr`
/// reflows this `next`, the same role `ListItem.read_locations` plays for
/// `listdef.read_item`.
fn shaped_array_getanyitem(inst: &super::model::SomeInstance) -> SomeValue {
    let Some(classdef) = inst.classdef.clone() else {
        panic!("getanyitem: array instance has no classdef");
    };
    let name = classdef.borrow().name.clone();
    if !majit_ir::descr::is_shaped_array_name(&name) {
        panic!("getanyitem: unsupported container instance {name}");
    }
    let fields: Vec<String> = {
        let borrowed = classdef.borrow();
        let mut names: Vec<String> = borrowed
            .attrs
            .keys()
            .filter(|k| k.starts_with("__pos_"))
            .cloned()
            .collect();
        names.sort_by_key(|n| {
            n.strip_prefix("__pos_")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(usize::MAX)
        });
        names
    };
    if fields.is_empty() {
        return s_impossible_value();
    }
    let classdesc = classdef.borrow().classdesc.clone();
    if let Some(bk) = classdef.borrow().bookkeeper.upgrade() {
        for field in &fields {
            let _ = bk.record_getattr(&classdesc, field);
        }
    }
    let mut items = Vec::with_capacity(fields.len());
    for field in &fields {
        match super::classdesc::ClassDef::s_getattr(&classdef, field, &inst.flags) {
            Ok(s) => items.push(s),
            Err(_) => items.push(s_impossible_value()),
        }
    }
    super::model::unionof(items.iter()).unwrap_or_else(|_| s_impossible_value())
}

// unaryop.py — class __extend__(SomePBC)

fn init_somepbc_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — setattr raises.
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::PBC,
        Specialization {
            apply: pure(|_ann, _hl| {
                panic!("AnnotatorError: Cannot modify attribute of a pre-built constant")
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — len raises.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::PBC,
        Specialization {
            apply: pure(|_ann, _hl| panic!("AnnotatorError: Cannot call len on a pbc")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// lltype.py — SomePtr.bool

fn init_someptr_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    fn with_ptr_like<R>(
        s: &SomeValue,
        f: impl FnOnce(&crate::annotator::model::SomePtr) -> R,
        g: impl FnOnce(&crate::annotator::model::SomeInteriorPtr) -> R,
    ) -> R {
        match s {
            SomeValue::Ptr(ptr) => f(ptr),
            SomeValue::InteriorPtr(ptr) => g(ptr),
            other => unreachable!("Ptr-tag dispatch must carry ptr-like value, got {other:?}"),
        }
    }
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Ptr,
        Specialization {
            apply: pure(|ann, hl| {
                let s = ann
                    .annotation(&hl.args[0])
                    .expect("ptr.getattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("ptr.getattr: attr unbound");
                with_ptr_like(
                    &s,
                    |ptr| ptr.getattr(&s_attr).expect("ptr.getattr must succeed"),
                    |ptr| {
                        ptr.getattr(&s_attr)
                            .expect("interiorptr.getattr must succeed")
                    },
                )
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // llmemory.py — SomeAddress.getattr(self, s_attr).
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Address,
        Specialization {
            apply: pure(|ann, hl| {
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("address.getattr: attr unbound");
                let attr = s_attr
                    .const_()
                    .and_then(ConstValue::as_pystr)
                    .expect("getattr on address must be constant str");
                SomeValue::TypedAddressAccess(
                    crate::translator::rtyper::lltypesystem::llmemory::SomeAddress::annotation_getattr(attr)
                        .unwrap_or_else(|| panic!("unsupported access type: {attr}")),
                )
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // llmemory.py — SomeAddress.bool(self) → s_Bool.
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Address,
        Specialization {
            apply: pure(|_ann, _hl| {
                crate::translator::rtyper::lltypesystem::llmemory::SomeAddress::annotation_bool()
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Ptr,
        Specialization {
            apply: pure(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("ptr.len: unbound");
                with_ptr_like(
                    &s,
                    |ptr| ptr.len().expect("ptr.len must succeed"),
                    |ptr| ptr.len().expect("interiorptr.len must succeed"),
                )
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Ptr,
        Specialization {
            apply: pure(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("ptr.bool: unbound");
                with_ptr_like(&s, |ptr| ptr.bool(), |ptr| ptr.bool())
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::Ptr,
        Specialization {
            // lltype.py — SomePtr.setattr "just doing checking",
            // returns None (void).
            apply: Box::new(|ann, hl| {
                let s = ann
                    .annotation(&hl.args[0])
                    .expect("ptr.setattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("ptr.setattr: attr unbound");
                let s_value = ann
                    .annotation(&hl.args[2])
                    .expect("ptr.setattr: value unbound");
                with_ptr_like(
                    &s,
                    |ptr| {
                        ptr.setattr(&s_attr, &s_value)
                            .expect("ptr.setattr must succeed")
                    },
                    |ptr| {
                        ptr.setattr(&s_attr, &s_value)
                            .expect("interiorptr.setattr must succeed")
                    },
                );
                None
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeNone)

fn init_somenone_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — getattr: s_ImpossibleValue, can_only_throw=[].
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::None_,
        Specialization {
            apply: pure(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — setattr: return None (void).
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| None),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py — call / simple_call / call_args: s_ImpossibleValue.
    for op in &[OpKind::SimpleCall, OpKind::CallArgs] {
        register(
            reg,
            *op,
            SomeValueTag::None_,
            Specialization {
                apply: pure(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py — len(self): SomeImpossibleValue.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::None_,
        Specialization {
            apply: pure(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeInstance)

fn init_someinstance_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // Shaped `Array<T;N>` only. `iter_SomeInstance` declines the
    // `__iter__` rewrite for that class so this arm runs: `SomeIterator`
    // of the array, matching `SomeList.iter`. Any other instance keeps
    // the transform and never reaches here.
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Instance,
        Specialization {
            apply: pure(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("instance.iter: unbound");
                if instance_is_shaped_array(&sv) {
                    SomeValue::Iterator(SomeIterator::new(sv, vec![]))
                } else {
                    s_impossible_value()
                }
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — SomeInstance.getattr(self, s_attr).
    //
    //   def getattr(self, s_attr):
    //       if not(s_attr.is_constant() and isinstance(s_attr.const, str)):
    //           raise AnnotatorError("A variable argument to getattr is not RPython")
    //       attr = s_attr.const
    //       if attr == '__class__':
    //           return self.classdef.read_attr__class__()
    //       getbookkeeper().record_getattr(self.classdef.classdesc, attr)
    //       return self.classdef.s_getattr(attr, self.flags)
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Instance,
        Specialization {
            apply: pure(|ann, hl| {
                let s_self = ann
                    .annotation(&hl.args[0])
                    .expect("instance.getattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("instance.getattr: attr unbound");
                let inst = match &s_self {
                    SomeValue::Instance(inst) => inst.clone(),
                    other => panic!(
                        "AnnotatorError: SomeInstance.getattr dispatched on non-instance: {:?}",
                        other
                    ),
                };
                let attr = s_attr
                    .const_()
                    .and_then(ConstValue::as_pystr)
                    .map(str::to_owned)
                    .unwrap_or_else(|| {
                        panic!(
                            "AnnotatorError: A variable argument to getattr is not RPython: {:?}",
                            s_attr
                        )
                    });
                // A raw-pointer inherent method is answered the way
                // `SomePtr.getattr` would: `is_null` resolves to a
                // bound method (lltype `_ptr.is_null`) whose call
                // yields `SomeBool`.  The frontend lowers
                // `p.is_null()` as `getattr(p, "is_null")` +
                // `simple_call(bound, …)` (`flowspace_adapter.rs`
                // CallTarget::Method), so getattr must return the
                // callable bound method — returning a bare `SomeBool`
                // would seat a non-callable in the simple_call callee
                // slot.  Upstream keeps this on `SomePtr.getattr`
                // (lltype.py) and `SomeInstance.getattr`
                // (unaryop.py) never sees it; pyre carries both
                // erased pointers (classdef-less) and `cast_pointer`-
                // typed receivers (`SomeInstance(cd)` from `obj as
                // *const W_Foo`) as `SomeInstance`, so the ptr method
                // answers only when the class hierarchy does not
                // itself define `is_null` — a real member keeps
                // upstream's classdef dispatch below.  An ordinary
                // (non-pointer) instance cannot reach this arm:
                // `getattr(recv, "is_null")` is only lowered from
                // `.is_null()` callsites, which rustc typechecks
                // against raw-pointer receivers or against a struct
                // member (and a member is found by the lookup guard).
                if attr == "is_null" {
                    let class_defines_attr = inst.classdef.as_ref().is_some_and(|cd| {
                        let classdesc = cd.borrow().classdesc.clone();
                        super::classdesc::ClassDesc::lookup(&classdesc, "is_null").is_some()
                    });
                    if !class_defines_attr {
                        return SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
                            "ptr_method_is_null",
                            s_self.clone(),
                            "is_null",
                        ));
                    }
                }
                if attr == "as_ref" || attr == "as_mut" {
                    let class_defines_attr = inst.classdef.as_ref().is_some_and(|cd| {
                        let classdesc = cd.borrow().classdesc.clone();
                        super::classdesc::ClassDesc::lookup(&classdesc, &attr).is_some()
                    });
                    if !class_defines_attr {
                        return SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
                            "ptr_method_as_ref",
                            s_self.clone(),
                            attr,
                        ));
                    }
                }
                // `to_exc_object` — `PyError`'s exception-object
                // materialization (error.rs `to_exc_object`), the
                // `rpyexc_raise` analog (exceptiontransform.py).  The
                // result_exc `?`-lowering mints it as a
                // `Method{receiver_root: "PyError"}` residual (result_exc.rs)
                // the codewriter resolves through `for_impl_method` to the real
                // conversion graph; but `PyError`'s synthesized class dict
                // carries no inherent-impl methods, so the `s_getattr` below
                // would Block.  Surface it as a residual builtin method
                // (mirroring the `is_null` ptr-method arm) so the annotator
                // types the boxed `W_BaseException` result while the codewriter
                // keeps the real fnaddr conversion.
                if attr == "to_exc_object" {
                    // Fires for a `classdef=Some(PyError)` receiver and for a
                    // classdef-less one.  `to_exc_object` is the sole method of
                    // that name (error.rs `PyError::to_exc_object`); it is only
                    // lowered from `.to_exc_object()` callsites rustc typechecks
                    // against a `PyError` receiver and from the result_exc
                    // `?`-lowering (`Method{receiver_root: "PyError"}`,
                    // result_exc.rs), so a classdef-less receiver here is a
                    // pointer-erased `PyError` call-result — `OpKind::Call`
                    // stamps a struct return `Ref(None)` with no `class_root`,
                    // unlike the seeded `Input` arm, so a call-returned `PyError`
                    // lifts `SomeInstance(classdef=None)`.  Same rustc-typecheck
                    // reasoning as the `is_null` ptr-method arm above.  The
                    // builtin result (`PyObjectRef`) and the codewriter residual
                    // (`for_impl_method("PyError", "to_exc_object")`) are both
                    // fixed, reading nothing off the receiver's classdef.
                    let receiver_is_pyerror = match inst.classdef.as_ref() {
                        Some(cd) => cd.borrow().name.rsplit('.').next() == Some("PyError"),
                        None => true,
                    };
                    let class_defines_attr = inst.classdef.as_ref().is_some_and(|cd| {
                        let classdesc = cd.borrow().classdesc.clone();
                        super::classdesc::ClassDesc::lookup(&classdesc, "to_exc_object").is_some()
                    });
                    if receiver_is_pyerror && !class_defines_attr {
                        return SomeValue::BuiltinMethod(SomeBuiltinMethod::new(
                            "pyerror_method_to_exc_object",
                            s_self.clone(),
                            "to_exc_object",
                        ));
                    }
                }
                let classdef = match inst.classdef.as_ref() {
                    Some(cd) => cd,
                    None => {
                        // `__discriminant` — synthetic enum-tag pseudo-attribute
                        // the MIR front-end's `Rvalue::Discriminant` lowering
                        // emits as a `FieldRead("__discriminant", ty: Int)`
                        // (`front/mir.rs`'s `build_rvalue`).  No RPython analogue (RPython
                        // source has no Rust enums); the tag is a low-level
                        // non-negative integer, declared `ValueType::Int` at the
                        // producer and trusted as such by the legacy walker
                        // (`legacy_annotator.rs`'s `infer_op_type`).  Resolve it here only
                        // for a
                        // classdef-less enum value (pyre's pointer erasure lifts
                        // an unseeded `Ref(_)` enum to a classdef-less
                        // `SomeInstance`); an enum receiver that carries its base
                        // `ClassDef` instead reads `__discriminant` through the
                        // ordinary `s_getattr` path below, where the Model A flat
                        // class projects the `i64` tag row to `SomeInteger`
                        // (`front/mir.rs`'s `derive_program_metadata`).  Keeping the
                        // short-circuit out of
                        // the classdef-bearing path mirrors RPython
                        // `SomeInstance.getattr`, which special-cases only
                        // `__class__` (unaryop.py).
                        if attr == "__discriminant" {
                            return SomeValue::Integer(super::model::SomeInteger::new(true, false));
                        }
                        // A classdef-less `SomeInstance` has no RPython analogue
                        // (`SomeInstance` upstream always carries a classdef).
                        // The mint is `valuetype_to_someshell`'s `Ref(_)` arm
                        // (`annotation_state.rs` `ref_fallback_instance`):
                        // `legacy_annotator::setbinding` writes that shell
                        // into `Variable.annotation` for every reference-typed
                        // variable, discarding any `Ref(Some(root))` payload.
                        // `project_struct_field_type` does not mint this for
                        // an unregistered named root — that arm returns
                        // `SomeValue::Impossible`.  It mints a classdef-less
                        // `SomeInstance` only for a raw-pointer-to-scalar/unit
                        // field and for the `"BigInt"` arm.  Any attr other
                        // than the `is_null` ptr method above stays fail-loud.
                        panic!(
                            "AnnotatorError: SomeInstance.getattr({:?}) on classdef-less instance",
                            attr
                        )
                    }
                };
                // unaryop.py:838-839 — `if attr == '__class__':
                //                          return self.classdef.read_attr__class__()`.
                if attr == "__class__" {
                    return super::classdesc::ClassDef::read_attr__class__(classdef);
                }
                // unaryop.py:840 — `getbookkeeper().record_getattr(
                //                       self.classdef.classdesc, attr)`.
                let classdesc = classdef.borrow().classdesc.clone();
                ann.bookkeeper
                    .record_getattr(&classdesc, &attr)
                    .unwrap_or_else(|err| {
                        panic!(
                            "AnnotatorError: SomeInstance.getattr({attr:?}) \
                             record_getattr failed: {err:?}"
                        )
                    });
                // unaryop.py — `return self.classdef.s_getattr(attr, self.flags)`.
                let s_attr = super::classdesc::ClassDef::s_getattr(classdef, &attr, &inst.flags)
                    .unwrap_or_else(|err| {
                        panic!("AnnotatorError: SomeInstance.s_getattr({attr:?}) failed: {err:?}")
                    });
                // `__discriminant` (the synthetic enum-tag pseudo-attribute;
                // see the classdef-less arm above) carries discriminant→variant
                // narrowing knowntypedata keyed by the receiver variable, so a
                // `match disc { k => ... }` switch refines this base-classed
                // receiver to SomeInstance(variant_k) per arm via follow_link's
                // improve.  No RPython analogue (RPython has no Rust enums).
                //
                // The tag is a non-negative integer.  An interpreter-defined
                // enum projects a `__discriminant` row that `s_getattr` returns
                // as `SomeInteger`; a per-instantiation `Option`/`Result`
                // classdef (#100) carries no row (the std enum was never
                // registered as a struct, so the bare template has nothing to
                // project) yet — unlike before the split — is classdef-bearing,
                // so it skips the classdef-less short-circuit above and reaches
                // here with `s_attr == Impossible`.  Synthesize the tag in that
                // case, the same value the classdef-less arm returns, so the
                // read does not block the inference.
                if attr == "__discriminant" {
                    let mut si = match &s_attr {
                        SomeValue::Integer(si) => si.clone(),
                        _ => super::model::SomeInteger::new(true, false),
                    };
                    if let Hlvalue::Variable(recv) = &hl.args[0] {
                        let enum_root = classdef.borrow().name.clone();
                        match ann.bookkeeper.enum_variant_narrowing_knowntypedata(
                            &enum_root,
                            &Rc::new(recv.clone()),
                        ) {
                            Ok(Some(ktd)) => si.set_knowntypedata(ktd),
                            Ok(None) => {}
                            Err(err) => panic!(
                                "AnnotatorError: SomeInstance.getattr({attr:?}) \
                                 enum variant knowntypedata failed: {err:?}"
                            ),
                        }
                    }
                    return SomeValue::Integer(si);
                }
                s_attr
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py — SomeInstance.setattr.
    //
    //   def setattr(self, s_attr, s_obj):
    //       if s_attr.is_constant() and isinstance(s_attr.const, str):
    //           attr = s_attr.const
    //           clsdef = self.classdef.locate_attribute(attr)
    //           attrdef = clsdef.attrs[attr]
    //           attrdef.modified(clsdef)
    //           if attrdef.s_value.contains(s_obj):
    //               return
    //           clsdef.generalize_attr(attr, s_obj)
    //           if isinstance(s_obj, SomeList):
    //               clsdef.classdesc.maybe_return_immutable_list(attr, s_obj)
    //       else:
    //           raise AnnotatorError("setattr(instance, variable_attr, value)")
    //
    // The success path records the write and falls through with no
    // return → None (void); a non-constant attr raises. (The base
    // `SomeObject.setattr` at unaryop.py is the one that returns
    // `s_ImpossibleValue`; this `SomeInstance` override does not.)
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::Instance,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = ann
                    .annotation(&hl.args[0])
                    .expect("instance.setattr: self unbound");
                let s_attr = ann
                    .annotation(&hl.args[1])
                    .expect("instance.setattr: attr unbound");
                let s_obj = ann
                    .annotation(&hl.args[2])
                    .expect("instance.setattr: value unbound");
                let inst = match &s_self {
                    SomeValue::Instance(inst) => inst.clone(),
                    other => panic!(
                        "AnnotatorError: SomeInstance.setattr dispatched on non-instance: {:?}",
                        other
                    ),
                };
                // unaryop.py:845,858 — `if s_attr.is_constant() and
                // isinstance(s_attr.const, str): … else: raise`.
                let attr = s_attr
                    .const_()
                    .and_then(ConstValue::as_pystr)
                    .map(str::to_owned)
                    .unwrap_or_else(|| {
                        panic!(
                            "AnnotatorError: setattr(instance, variable_attr, value): {:?}",
                            s_attr
                        )
                    });
                let classdef = match inst.classdef.as_ref() {
                    Some(cd) => cd.clone(),
                    None => {
                        // A classdef-less `SomeInstance` (pyre pointer erasure;
                        // see the getattr arm above) has no `ClassDef` to record
                        // the write on.  Stays fail-loud like getattr's non-
                        // `is_null` arm — a cast-narrowed receiver carries its
                        // struct-root classdef (front-end pointer-downcast carry),
                        // so a writable instance reaches here with a populated
                        // classdef.
                        panic!(
                            "AnnotatorError: SomeInstance.setattr({:?}) on classdef-less instance",
                            attr
                        )
                    }
                };
                // unaryop.py — `clsdef = self.classdef.locate_attribute(attr)`.
                let clsdef = super::classdesc::ClassDef::locate_attribute(&classdef, &attr)
                    .unwrap_or_else(|err| {
                        panic!(
                            "AnnotatorError: SomeInstance.setattr({attr:?}) \
                             locate_attribute failed: {err:?}"
                        )
                    });
                // unaryop.py:848-852 — `attrdef = clsdef.attrs[attr];
                // attrdef.modified(clsdef); if attrdef.s_value.contains(s_obj):
                // return`.
                // `attrdef.modified` needs `clsdef` mutably, but the following
                // `s_value.contains(s_obj)` reads `clsdef`'s mro through
                // `commonbase` (a shared borrow) when either side carries a
                // classdef — so the mutable borrow must be released first.
                // Upstream (unaryop.py:848-852) holds no exclusive borrow;
                // clone `s_value` out and drop the borrow before the read.
                let s_value = {
                    let mut clsdef_mut = clsdef.borrow_mut();
                    let attrdef = clsdef_mut.attrs.get_mut(&attr).unwrap_or_else(|| {
                        panic!(
                            "AnnotatorError: SomeInstance.setattr({attr:?}) \
                             attr absent after locate_attribute"
                        )
                    });
                    attrdef.modified(Some(&clsdef)).unwrap_or_else(|err| {
                        panic!(
                            "AnnotatorError: SomeInstance.setattr({attr:?}) \
                             modified failed: {err:?}"
                        )
                    });
                    attrdef.s_value.clone()
                };
                let already_contains = s_value.contains(&s_obj);
                if !already_contains {
                    // unaryop.py — `clsdef.generalize_attr(attr, s_obj)`.
                    super::classdesc::ClassDef::generalize_attr(
                        &clsdef,
                        &attr,
                        Some(s_obj.clone()),
                    )
                    .unwrap_or_else(|err| {
                        panic!(
                            "AnnotatorError: SomeInstance.setattr({attr:?}) \
                             generalize_attr failed: {err:?}"
                        )
                    });
                    // unaryop.py:856-857 — `if isinstance(s_obj, SomeList):
                    // clsdef.classdesc.maybe_return_immutable_list(attr, s_obj)`.
                    if matches!(s_obj, SomeValue::List(_)) {
                        let classdesc = clsdef.borrow().classdesc.clone();
                        super::classdesc::ClassDesc::maybe_return_immutable_list(
                            &classdesc, &attr, &s_obj,
                        )
                        .unwrap_or_else(|err| {
                            panic!(
                                "AnnotatorError: SomeInstance.setattr({attr:?}) \
                                 maybe_return_immutable_list failed: {err:?}"
                            )
                        });
                    }
                }
                // unaryop.py setattr — success path returns None (void).
                None
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// unaryop.py — class __extend__(SomeWeakRef)

fn init_someweakref_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py — simple_call(self):
    //   if classdef is None: return s_None (dead weakref)
    //   else: SomeInstance(classdef, can_be_None=True)
    register(
        reg,
        OpKind::SimpleCall,
        SomeValueTag::WeakRef,
        Specialization {
            apply: pure(|ann, hl| {
                let wr = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::WeakRef(w)) => w,
                    _ => panic!("weakref.simple_call: arg 0 not SomeWeakRef"),
                };
                match &wr.classdef {
                    None => super::model::s_none(),
                    Some(cd) => SomeValue::Instance(super::model::SomeInstance::new(
                        Some(Rc::clone(cd)),
                        true,
                        std::collections::BTreeMap::new(),
                    )),
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// Suppress unused warnings for items reserved for the follow-up
// commits in the same family (SomeUnicodeString, SomeBool
// const-propagation paths, Rc).
#[allow(dead_code)]
fn _keep_imports_live() {
    let _ = SomeUnicodeString::default;
    let _: Option<Rc<()>> = None;
}

// unaryop.py — @op.call_args.register_transform(SomeObject)
//
// Upstream rewrites `op.call_args(v_func, v_shape, *data_v)` when the
// vararg tail is a `SomeTuple`, unpacking the tuple elements to turn
// the op back into `op.simple_call(...)` (or `op.call_args(...)` if
// keywords remain). The Rust port requires `CallShape` decoding from
// the `ConstValue::Tuple` payload on `v_shape` (mirrors
// `CallSpec.fromshape`).

pub fn transform_varargs(ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    use super::super::flowspace::argument::CallSpec;

    // args = [v_func, v_shape, *data_v]
    if args.len() < 2 {
        return None;
    }
    let v_func = args[0].clone();
    let v_shape = &args[1];
    // upstream: `CallSpec.fromshape(v_shape.value, list(data_v))`.
    let shape_const = match v_shape {
        Hlvalue::Constant(c) => &c.value,
        _ => return None,
    };
    let shape = decode_call_shape(shape_const)?;
    let data_v: Vec<Hlvalue> = args[2..].to_vec();
    let callspec = CallSpec::fromshape(&shape, data_v);
    // upstream: `v_vararg = callspec.w_stararg; if callspec.w_stararg: ...`.
    let v_vararg = callspec.w_stararg.clone()?;
    // upstream: `s_vararg = annotator.annotation(v_vararg)`.
    let s_vararg = ann.annotation(&v_vararg)?;
    // upstream: `if not isinstance(s_vararg, SomeTuple): raise AnnotatorError`.
    let super::model::SomeValue::Tuple(ref tup) = s_vararg else {
        panic!("AnnotatorError: Calls like f(..., *arg) require 'arg' to be a tuple");
    };
    let n_items = tup.items.len();
    // upstream: `ops = [op.getitem(v_vararg, const(i)) for i in range(n_items)]`.
    let mut ops: Vec<HLOperation> = Vec::with_capacity(n_items + 1);
    let mut unpacked: Vec<Hlvalue> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let getitem = mk_hlop(
            OpKind::GetItem,
            vec![
                v_vararg.clone(),
                Hlvalue::Constant(Constant::new(ConstValue::Int(i as i64))),
            ],
        );
        unpacked.push(Hlvalue::Variable(getitem.result.clone()));
        ops.push(getitem);
    }
    // upstream: `new_args = callspec.arguments_w + [hlop.result for ...]`.
    let mut new_args: Vec<Hlvalue> = callspec.arguments_w.clone();
    new_args.extend(unpacked);
    // upstream: `if callspec.keywords: newspec = CallSpec(new_args, callspec.keywords);
    //            shape, data_v = newspec.flatten(); op.call_args(v_func, const(shape), *data_v)`
    //           else: `op.simple_call(v_func, *new_args)`.
    let call_op = if !callspec.keywords.is_empty() {
        let newspec = CallSpec::new(new_args, Some(callspec.keywords), None);
        let (new_shape, new_data_v) = newspec.flatten();
        let mut call_args = vec![
            v_func,
            Hlvalue::Constant(Constant::new(encode_call_shape(&new_shape))),
        ];
        call_args.extend(new_data_v);
        mk_hlop(OpKind::CallArgs, call_args)
    } else {
        let mut simple_args = vec![v_func];
        simple_args.extend(new_args);
        mk_hlop(OpKind::SimpleCall, simple_args)
    };
    ops.push(call_op);
    Some(ops)
}

fn init_object_call_args_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    // unaryop.py — @op.call_args.register_transform(SomeObject).
    // ```python
    // def transform_varargs(annotator, v_func, v_shape, *data_v):
    //     callspec = CallSpec.fromshape(v_shape.value, list(data_v))
    //     v_vararg = callspec.w_stararg
    //     if callspec.w_stararg:
    //         s_vararg = annotator.annotation(callspec.w_stararg)
    //         if not isinstance(s_vararg, SomeTuple):
    //             raise AnnotatorError(...)
    //         n_items = len(s_vararg.items)
    //         ops = [op.getitem(v_vararg, const(i)) for i in range(n_items)]
    //         new_args = callspec.arguments_w + [hlop.result for hlop in ops]
    //         if callspec.keywords:
    //             newspec = CallSpec(new_args, callspec.keywords)
    //             shape, data_v = newspec.flatten()
    //             call_op = op.call_args(v_func, const(shape), *data_v)
    //         else:
    //             call_op = op.simple_call(v_func, *new_args)
    //         ops.append(call_op)
    //         return ops
    // ```
    register_transform(
        reg,
        OpKind::CallArgs,
        SomeValueTag::Object,
        Box::new(transform_varargs),
    );
}

/// Decode the `CallShape` triple encoded in a `call_args` shape
/// constant (see `flowcontext::build_call_shape_constant` for the
/// encoder — ConstValue::Tuple([Int(cnt), Tuple([Str(key), …]), Bool(star)])).
pub(crate) fn decode_call_shape(
    cv: &ConstValue,
) -> Option<super::super::flowspace::argument::CallShape> {
    let items = match cv {
        ConstValue::Tuple(items) if items.len() == 3 => items,
        _ => return None,
    };
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        _ => return None,
    };
    let shape_keys: Vec<String> = match &items[1] {
        ConstValue::Tuple(keys) => keys
            .iter()
            .map(|value| value.as_text().map(str::to_owned))
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        _ => return None,
    };
    Some(super::super::flowspace::argument::CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
}

/// Encode a `CallShape` as the `ConstValue::Tuple` payload expected by
/// `call_args` — mirrors `flowcontext::build_call_shape_constant`.
fn encode_call_shape(shape: &super::super::flowspace::argument::CallShape) -> ConstValue {
    ConstValue::Tuple(vec![
        ConstValue::Int(shape.shape_cnt as i64),
        ConstValue::Tuple(shape.shape_keys.iter().map(ConstValue::byte_str).collect()),
        ConstValue::Bool(shape.shape_star),
    ])
}

// unaryop.py — @op.{len,iter,next,getslice,setslice}.register_transform(SomeInstance)

/// Helper — build a new [`HLOperation`] with a fresh result variable,
/// matching upstream `HLOperation.__init__` (operation.py:73-75).
fn mk_hlop(kind: OpKind, args: Vec<Hlvalue>) -> HLOperation {
    HLOperation::new(kind, args)
}

/// RPython `_find_property_meth(s_obj, attr, meth)` (unaryop.py).
///
/// Walks `s_obj.classdef.getmro()` and collects the `property` descriptor's
/// `fget` / `fset` / `fdel` method reference for `attr`. Returns `None`
/// when the attribute appears in any classdict entry that isn't a
/// `property` (mirrors upstream's `return` without a result).
///
/// `meth` selects which property slot to read — "fget" / "fset" / "fdel" —
/// matching upstream's `getattr(obj.value, meth)`.
fn find_property_meth(
    s_obj: &super::model::SomeInstance,
    attr: &str,
    meth: &str,
) -> Option<Vec<Option<super::super::flowspace::model::HostObject>>> {
    use super::super::flowspace::model::ConstValue;
    let classdef = s_obj.classdef.as_ref()?;
    let mro = super::classdesc::ClassDef::getmro(classdef);
    let mut result: Vec<Option<super::super::flowspace::model::HostObject>> = Vec::new();
    for clsdef in &mro {
        let classdesc = clsdef.borrow().classdesc.clone();
        let entry = classdesc.borrow().classdict.get(attr).cloned();
        let Some(entry) = entry else { continue };
        match entry {
            super::classdesc::ClassDictEntry::Constant(c) => {
                // upstream unaryop.py:902-904:
                //   if (not isinstance(obj, Constant) or
                //           not isinstance(obj.value, property)):
                //       return
                //   result.append(getattr(obj.value, meth))
                let ConstValue::HostObject(host) = &c.value else {
                    return None;
                };
                if !host.is_property() {
                    return None;
                }
                let slot = match meth {
                    "fget" => host.property_fget().cloned(),
                    "fset" => host.property_fset().cloned(),
                    "fdel" => host.property_fdel().cloned(),
                    _ => return None,
                };
                result.push(slot);
            }
            // Non-Constant classdict entries (FunctionDesc etc.) never
            // stand in for a property; upstream's isinstance-check falls
            // through to `return`.
            super::classdesc::ClassDictEntry::Desc(_) => return None,
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

#[allow(non_snake_case)]
pub fn len_SomeInstance(_ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_arg = args[0].clone();
    let get_len = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_arg,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__len__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_len.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
    Some(vec![get_len, call])
}

#[allow(non_snake_case)]
pub fn iter_SomeInstance(ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    // `Array<T;N>` is iterated as a container: `iter` yields
    // `SomeIterator(self)` and `next` reads the element via `getanyitem`
    // (`SomeIterator.next`). Rewriting to `__iter__` would type the
    // element from the residual return shell (`Ref` → classdef-less
    // instance) instead of the constructor's item annotation.
    if ann
        .annotation(&args[0])
        .is_some_and(|s| instance_is_shaped_array(&s))
    {
        return None;
    }
    let v_arg = args[0].clone();
    let get_iter = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_arg,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__iter__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_iter.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
    Some(vec![get_iter, call])
}

#[allow(non_snake_case)]
pub fn next_SomeInstance(_ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_arg = args[0].clone();
    let get_next = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_arg,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("next"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_next.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
    Some(vec![get_next, call])
}

#[allow(non_snake_case)]
pub fn getslice_SomeInstance(
    _ann: &RPythonAnnotator,
    args: &[Hlvalue],
) -> Option<Vec<HLOperation>> {
    let v_obj = args[0].clone();
    let v_start = args[1].clone();
    let v_stop = args[2].clone();
    let get_getslice = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_obj,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__getslice__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_getslice.result.clone());
    let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_start, v_stop]);
    Some(vec![get_getslice, call])
}

#[allow(non_snake_case)]
pub fn setslice_SomeInstance(
    _ann: &RPythonAnnotator,
    args: &[Hlvalue],
) -> Option<Vec<HLOperation>> {
    let v_obj = args[0].clone();
    let v_start = args[1].clone();
    let v_stop = args[2].clone();
    let v_iterable = args[3].clone();
    let get_setslice = mk_hlop(
        OpKind::GetAttr,
        vec![
            v_obj,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("__setslice__"))),
        ],
    );
    let getattr_result = Hlvalue::Variable(get_setslice.result.clone());
    let call = mk_hlop(
        OpKind::SimpleCall,
        vec![getattr_result, v_start, v_stop, v_iterable],
    );
    Some(vec![get_setslice, call])
}

fn init_instance_single_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    super::binaryop::init_contains_instance_transform(reg);

    // unaryop.py — len(v_arg) -> [getattr(v_arg, '__len__'), simple_call(getattr.result)]
    register_transform(
        reg,
        OpKind::Len,
        SomeValueTag::Instance,
        Box::new(len_SomeInstance),
    );
    // unaryop.py — iter
    register_transform(
        reg,
        OpKind::Iter,
        SomeValueTag::Instance,
        Box::new(iter_SomeInstance),
    );
    // unaryop.py — next
    register_transform(
        reg,
        OpKind::Next,
        SomeValueTag::Instance,
        Box::new(next_SomeInstance),
    );
    // unaryop.py — getslice
    register_transform(
        reg,
        OpKind::GetSlice,
        SomeValueTag::Instance,
        Box::new(getslice_SomeInstance),
    );
    // unaryop.py — setslice
    register_transform(
        reg,
        OpKind::SetSlice,
        SomeValueTag::Instance,
        Box::new(setslice_SomeInstance),
    );
}

// unaryop.py — @op.{getattr,setattr}.register_transform(SomeInstance)

/// RPython `getattr_SomeInstance` / `setattr_SomeInstance`
/// (unaryop.py) — property descriptor dispatch.
///
/// Upstream rewrites `getattr(v_obj, 'prop')` into
/// `[getattr(v_obj, 'prop__getter__'), simple_call(getter.result)]`
/// when every classdef in the MRO has `prop` bound as a `property`
/// descriptor, and raises `AnnotatorError` when none of them do (the
/// "mixed property/non-property" case falls through with no rewrite).
/// `setattr` is the setter analogue — three-arg `simple_call(setter,
/// v_value)` instead.
#[allow(non_snake_case)]
pub fn getattr_SomeInstance(ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_obj = args[0].clone();
    let v_attr = &args[1];
    // upstream: `s_attr = annotator.annotation(v_attr)`.
    let s_attr = ann.annotation(v_attr)?;
    // upstream: `if not s_attr.is_constant() or not isinstance(s_attr.const, str): return`.
    // `isinstance(x, str)` is Python 2 bytes-only — narrow with
    // [`ConstValue::as_pystr`] to keep parity with the upstream
    // short-circuit on unicode constants.
    let attr = match s_attr.const_() {
        Some(value) => value.as_pystr().map(str::to_owned)?,
        _ => return None,
    };
    // upstream: `getters = _find_property_meth(annotator.annotation(v_obj), attr, 'fget')`.
    let s_obj = ann.annotation(&v_obj)?;
    let super::model::SomeValue::Instance(ref s_inst) = s_obj else {
        return None;
    };
    let getters = find_property_meth(s_inst, &attr, "fget")?;
    // upstream: `if all(getters): … elif not any(getters): raise`.
    if getters.iter().all(|g| g.is_some()) {
        let getter_name = format!("{attr}__getter__");
        let get_getter = mk_hlop(
            OpKind::GetAttr,
            vec![
                v_obj,
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(getter_name))),
            ],
        );
        let getter_result = Hlvalue::Variable(get_getter.result.clone());
        let call = mk_hlop(OpKind::SimpleCall, vec![getter_result]);
        Some(vec![get_getter, call])
    } else if !getters.iter().any(|g| g.is_some()) {
        panic!("AnnotatorError: Attribute {attr:?} is unreadable");
    } else {
        None
    }
}

#[allow(non_snake_case)]
pub fn setattr_SomeInstance(ann: &RPythonAnnotator, args: &[Hlvalue]) -> Option<Vec<HLOperation>> {
    let v_obj = args[0].clone();
    let v_attr = &args[1];
    let v_value = args[2].clone();
    let s_attr = ann.annotation(v_attr)?;
    let attr = match s_attr.const_() {
        Some(value) => value.as_pystr().map(str::to_owned)?,
        _ => return None,
    };
    let s_obj = ann.annotation(&v_obj)?;
    let super::model::SomeValue::Instance(ref s_inst) = s_obj else {
        return None;
    };
    let setters = find_property_meth(s_inst, &attr, "fset")?;
    if setters.iter().all(|s| s.is_some()) {
        let setter_name = format!("{attr}__setter__");
        let get_setter = mk_hlop(
            OpKind::GetAttr,
            vec![
                v_obj,
                Hlvalue::Constant(Constant::new(ConstValue::byte_str(setter_name))),
            ],
        );
        let setter_result = Hlvalue::Variable(get_setter.result.clone());
        let call = mk_hlop(OpKind::SimpleCall, vec![setter_result, v_value]);
        Some(vec![get_setter, call])
    } else if !setters.iter().any(|s| s.is_some()) {
        panic!("AnnotatorError: Attribute {attr:?} is unwritable");
    } else {
        None
    }
}

fn init_instance_attr_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    // unaryop.py — getattr
    register_transform(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Instance,
        Box::new(getattr_SomeInstance),
    );
    // unaryop.py — setattr_SomeInstance.
    //
    //   if not s_attr.is_constant() or not isinstance(s_attr.const, str): return
    //
    // Mirror getter's `isinstance(value, str)` Python 2 bytes-only
    // gate via [`ConstValue::as_pystr`] so a unicode literal short-
    // circuits the same way it does on the getattr side, instead of
    // dispatching against `_find_property_meth(..., 'fset')`.
    register_transform(
        reg,
        OpKind::SetAttr,
        SomeValueTag::Instance,
        Box::new(setattr_SomeInstance),
    );
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};
    use super::*;

    fn mk_ann() -> Rc<RPythonAnnotator> {
        RPythonAnnotator::new(None, None, None, false)
    }

    fn hl1(op: OpKind, sv: SomeValue) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = mk_ann();
        let mut v = Variable::named("x");
        ann.setbinding(&mut v, sv);
        let hl = HLOperation::new(op, vec![Hlvalue::Variable(v)]);
        (hl, ann)
    }

    #[test]
    fn unary_operations_list_excludes_contains() {
        assert!(!UNARY_OPERATIONS.contains(&OpKind::Contains));
        assert!(UNARY_OPERATIONS.contains(&OpKind::Len));
        assert!(UNARY_OPERATIONS.contains(&OpKind::Bool));
    }

    #[test]
    fn consider_isinstance_someobject_uses_known_type_subclass_refinement() {
        let ann = mk_ann();
        let mut v_obj = Variable::named("obj");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(&mut v_obj, SomeValue::Bool(SomeBool::new()));
        let s_cls = ann
            .bookkeeper
            .immutablevalue(&ConstValue::builtin("int"))
            .expect("immutablevalue(int)");
        ann.setbinding(&mut v_cls, s_cls);
        let hl = HLOperation::new(
            OpKind::IsInstance,
            vec![Hlvalue::Variable(v_obj), Hlvalue::Variable(v_cls)],
        );

        let result = hl.consider(&ann).expect("isinstance").unwrap();
        let SomeValue::Bool(b) = result else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(true));
    }

    #[test]
    fn consider_isinstance_plain_someobject_does_not_overfold_to_false() {
        let ann = mk_ann();
        let mut v_obj = Variable::named("obj");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(&mut v_obj, SomeValue::object());
        let s_cls = ann
            .bookkeeper
            .immutablevalue(&ConstValue::builtin("int"))
            .expect("immutablevalue(int)");
        ann.setbinding(&mut v_cls, s_cls);
        let hl = HLOperation::new(
            OpKind::IsInstance,
            vec![Hlvalue::Variable(v_obj), Hlvalue::Variable(v_cls)],
        );

        let result = hl.consider(&ann).expect("isinstance").unwrap();
        let SomeValue::Bool(b) = result else {
            panic!("expected SomeBool");
        };
        assert!(b.base.const_box.is_none());
    }

    #[test]
    fn consider_isinstance_user_class_named_list_is_not_builtin_list() {
        // `our_issubclass` compares class identities (`unaryop.py`).
        // Routing `"list"` through `KnownType::List` by qualname made a
        // user class named `list` look like builtin `list`.
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("list", None);
        let mut v_obj = Variable::named("obj");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let s_cls = ann
            .bookkeeper
            .immutablevalue(&ConstValue::builtin("list"))
            .expect("immutablevalue(list)");
        ann.setbinding(&mut v_cls, s_cls);
        let hl = HLOperation::new(
            OpKind::IsInstance,
            vec![Hlvalue::Variable(v_obj), Hlvalue::Variable(v_cls)],
        );

        let result = hl.consider(&ann).expect("isinstance").unwrap();
        let SomeValue::Bool(b) = result else {
            panic!("expected SomeBool");
        };
        assert_eq!(
            b.base.const_box.unwrap().value,
            ConstValue::Bool(false),
            "user class named list is not builtin list"
        );
    }

    #[test]
    fn consider_issubtype_sometypeof_routes_through_s_isinstance() {
        let ann = mk_ann();
        let mut v_obj = Variable::named("obj");
        let mut v_type = Variable::named("type_of_obj");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(&mut v_obj, SomeValue::Bool(SomeBool::new()));
        ann.setbinding(
            &mut v_type,
            SomeValue::TypeOf(SomeTypeOf::new(vec![Rc::new(v_obj.clone())])),
        );
        let s_cls = ann
            .bookkeeper
            .immutablevalue(&ConstValue::builtin("int"))
            .expect("immutablevalue(int)");
        ann.setbinding(&mut v_cls, s_cls);
        let hl = HLOperation::new(
            OpKind::IsSubtype,
            vec![Hlvalue::Variable(v_type), Hlvalue::Variable(v_cls)],
        );

        let result = hl.consider(&ann).expect("issubtype").unwrap();
        let SomeValue::Bool(b) = result else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(true));
    }

    #[test]
    fn consider_getattr_prefers_find_method_on_lists() {
        let ann = mk_ann();
        let mut v = Variable::named("lst");
        ann.setbinding(
            &mut v,
            SomeValue::List(super::super::model::SomeList::new(
                super::super::listdef::ListDef::new(
                    Some(ann.bookkeeper.clone()),
                    SomeValue::Integer(SomeInteger::default()),
                    false,
                    false,
                ),
            )),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("append"))),
            ],
        );

        let result = hl.consider(&ann).expect("getattr must succeed").unwrap();

        let SomeValue::BuiltinMethod(method) = result else {
            panic!("expected SomeBuiltinMethod");
        };
        assert_eq!(method.analyser_name, "list_method_append");
        assert_eq!(method.methodname, "append");
    }

    #[test]
    fn transform_len_someinstance_rewrites_to_getattr_plus_simple_call() {
        // unaryop.py — `op.len(v_ins)` → [getattr(v_ins, '__len__'), simple_call(r)].
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v = Variable::named("inst");
        ann.setbinding(
            &mut v,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let hl = HLOperation::new(OpKind::Len, vec![Hlvalue::Variable(v)]);
        let new_ops = hl
            .transform(&ann)
            .expect("Len(SomeInstance) must register_transform");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn transform_call_args_unpacks_tuple_stararg_to_simple_call() {
        // unaryop.py:120-139 — `f(*tup)` where tup has 3 items and no
        // keywords should rewrite to 3 getitem ops + one simple_call.
        use super::super::model::{SomeInteger, SomeTuple};
        let ann = mk_ann();
        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, SomeValue::object());
        let mut v_tup = Variable::named("tup");
        ann.setbinding(
            &mut v_tup,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
        );
        // `f(*tup)` encodes as: call_args(f, shape, tup) where shape =
        // (shape_cnt=0, shape_keys=(), shape_star=True).
        let shape_const = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(0),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(true),
        ])));
        let hl = HLOperation::new(
            OpKind::CallArgs,
            vec![
                Hlvalue::Variable(v_func),
                shape_const,
                Hlvalue::Variable(v_tup),
            ],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("call_args(SomeObject, *SomeTuple) must transform");
        // 3 getitem + 1 simple_call = 4 ops
        assert_eq!(new_ops.len(), 4);
        assert!(matches!(new_ops[0].kind, OpKind::GetItem));
        assert!(matches!(new_ops[1].kind, OpKind::GetItem));
        assert!(matches!(new_ops[2].kind, OpKind::GetItem));
        assert!(matches!(new_ops[3].kind, OpKind::SimpleCall));
        // simple_call receives v_func + 3 unpacked values
        assert_eq!(new_ops[3].args.len(), 4);
    }

    #[test]
    fn transform_call_args_without_stararg_is_noop() {
        // upstream: if callspec.w_stararg is None, transform_varargs
        // falls through without returning ops.
        let ann = mk_ann();
        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, SomeValue::object());
        let shape_const = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(0),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(false),
        ])));
        let hl = HLOperation::new(
            OpKind::CallArgs,
            vec![Hlvalue::Variable(v_func), shape_const],
        );
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_getattr_someinstance_without_property_is_noop() {
        // unaryop.py:909-921 — when the target attribute isn't a
        // property descriptor in any classdef, `_find_property_meth`
        // returns None and the transform leaves the op alone.
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::byte_str("prop")));
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_obj), v_attr_const],
        );
        // Registered but returns None because the classdict carries no
        // `property` descriptor (Rust HostObject property variant not
        // yet modelled).
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_setattr_someinstance_without_property_is_noop() {
        // unaryop.py:924-936 setter analogue.
        use super::super::classdesc::ClassDef;
        use super::super::model::{SomeInstance, SomeInteger};
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let mut v_value = Variable::named("val");
        ann.setbinding(&mut v_value, SomeValue::Integer(SomeInteger::default()));
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::byte_str("prop")));
        let hl = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_obj),
                v_attr_const,
                Hlvalue::Variable(v_value),
            ],
        );
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_getattr_someinstance_with_property_rewrites_via_getter_hidden_fn() {
        // unaryop.py:909-921 — when every classdef in the MRO binds
        // `prop` as a `property` descriptor, the transform rewrites
        // `getattr(v_obj, 'prop')` into `[getattr(v_obj, 'prop__getter__'),
        // simple_call(getter.result)]`.
        use super::super::super::flowspace::model::GraphFunc;
        use super::super::classdesc::{ClassDef, ClassDictEntry};
        use super::super::model::SomeInstance;
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.Y", None);
        // Build a property(fget, fset) and bind it as the `prop` entry
        // in the classdesc's classdict — simulating what
        // `ClassDesc::add_source_attribute` produces when processing a
        // property descriptor.
        let fget_host =
            super::super::super::flowspace::model::HostObject::new_user_function(GraphFunc::new(
                "get_prop",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
        let fset_host =
            super::super::super::flowspace::model::HostObject::new_user_function(GraphFunc::new(
                "set_prop",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
        let prop_host = super::super::super::flowspace::model::HostObject::new_property(
            "pkg.Y.prop",
            Some(fget_host),
            Some(fset_host),
            None,
        );
        let classdesc = classdef.borrow().classdesc.clone();
        classdesc.borrow_mut().classdict.insert(
            "prop".to_string(),
            ClassDictEntry::constant(ConstValue::HostObject(prop_host)),
        );
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::byte_str("prop")));
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_obj), v_attr_const],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("property attr must rewrite to getter+simple_call");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        // The rewritten getattr target is the hidden getter name.
        let target_attr = &new_ops[0].args[1];
        match target_attr {
            Hlvalue::Constant(c) => match &c.value {
                value if value.string_eq("prop__getter__") => {}
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant attr, got {other:?}"),
        }
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn transform_getitem_someinstance_rewrites_to_getattr_plus_simple_call() {
        // binaryop.py — `op.getitem(v_ins, v_idx)` → getattr + simple_call.
        use super::super::classdesc::ClassDef;
        use super::super::model::{SomeInstance, SomeInteger};

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_ins = Variable::named("inst");
        ann.setbinding(
            &mut v_ins,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let mut v_idx = Variable::named("i");
        ann.setbinding(&mut v_idx, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v_ins), Hlvalue::Variable(v_idx)],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("GetItem(SomeInstance, SomeObject) must register_transform");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn consider_someobject_len_returns_nonneg_integer() {
        // unaryop.py — default len returns SomeInteger(nonneg=True).
        let (hl, ann) = hl1(OpKind::Len, SomeValue::object());
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg, "nonneg not set"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somefloat_neg_returns_somefloat() {
        let (hl, ann) = hl1(OpKind::Neg, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somefloat_pos_returns_self() {
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_bool_on_none_is_constant_false() {
        let (hl, ann) = hl1(
            OpKind::Bool,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(false));
    }

    #[test]
    fn consider_someobject_bool_adds_truth_branch_nonnone_refinement() {
        let ann = mk_ann();
        let mut v = Variable::named("maybe_s");
        ann.setbinding(&mut v, SomeValue::String(SomeString::new(true, false)));
        let hl = HLOperation::new(OpKind::Bool, vec![Hlvalue::Variable(v.clone())]);
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        let narrowed = ktd
            .get(&ExitCaseKey::Bool(true))
            .and_then(|inner| inner.get(&Rc::new(v)))
            .expect("truth branch must refine the object");
        match narrowed {
            SomeValue::String(s) => assert!(!s.inner.can_be_none),
            other => panic!("expected SomeString refinement, got {other:?}"),
        }
    }

    /// `bool(disc)` of a two-variant discriminant narrows the receiver to
    /// the tag-0 variant on `False` and the tag-1 variant on `True`.
    #[test]
    fn bool_of_a_discriminant_keeps_the_variant_narrowing() {
        let recv = Rc::new(Variable::named("shell"));
        let s_ok = SomeValue::String(SomeString::new(false, false));
        let s_err = SomeValue::Integer(SomeInteger::default());
        let mut ktd = super::super::model::KnownTypeData::new();
        add_knowntypedata(
            &mut ktd,
            ExitCaseKey::Int(0),
            std::slice::from_ref(&recv),
            s_ok.clone(),
        );
        add_knowntypedata(
            &mut ktd,
            ExitCaseKey::Int(1),
            std::slice::from_ref(&recv),
            s_err.clone(),
        );
        let mut s_disc = SomeInteger::new(true, false);
        s_disc.set_knowntypedata(ktd);
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Integer(s_disc));
        let SomeValue::Bool(b) = hl.consider(&ann).unwrap().unwrap() else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        assert_eq!(ktd[&ExitCaseKey::Bool(false)][&recv], s_ok);
        assert_eq!(ktd[&ExitCaseKey::Bool(true)][&recv], s_err);
    }

    #[test]
    fn consider_someobject_bool_uses_constant_len_when_available() {
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Tuple(SomeTuple::new(vec![])));
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(false));
    }

    #[test]
    fn consider_someinteger_abs_is_nonneg() {
        let (hl, ann) = hl1(OpKind::Abs, SomeValue::Integer(SomeInteger::default()));
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somebool_bool_returns_self() {
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someptr_bool_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget, functionptr,
        };

        let ann = mk_ann();
        let ptr = functionptr(
            FuncType {
                args: vec![],
                result: LowLevelType::Void,
            },
            "f",
            None,
            None,
        );
        let mut s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        s_ptr.base.const_box = Some(Constant::new(ConstValue::LLPtr(Box::new(ptr))));
        let mut v = Variable::named("p");
        ann.setbinding(&mut v, SomeValue::Ptr(s_ptr));
        let hl = HLOperation::new(OpKind::Bool, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        assert_eq!(b.base.const_box.unwrap().value, ConstValue::Bool(true));
    }

    #[test]
    fn consider_someptr_len_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            Array, LowLevelType, Ptr, PtrTarget,
        };

        // upstream `SomePtr.len()` (lltype.py) propagates
        // `_fixedlength`'s `len()` probe, which raises `TypeError` on
        // non-array pointers. Use an Array ptr so the override returns
        // `SomeInteger(nonneg=True)` (the varsize-array path).
        let ann = mk_ann();
        let s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Array(Array::new(LowLevelType::Signed)),
        });
        let mut v = Variable::named("p");
        ann.setbinding(&mut v, SomeValue::Ptr(s_ptr));
        let hl = HLOperation::new(OpKind::Len, vec![Hlvalue::Variable(v)]);

        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Integer(i) = r else {
            panic!("expected SomeInteger");
        };
        assert!(i.nonneg);
    }

    #[test]
    fn consider_someinteriorptr_bool_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            InteriorOffset, InteriorPtr, LowLevelType, Struct,
        };

        let ann = mk_ann();
        let mut v = Variable::named("p");
        ann.setbinding(
            &mut v,
            SomeValue::InteriorPtr(super::super::model::SomeInteriorPtr::new(InteriorPtr {
                PARENTTYPE: Box::new(LowLevelType::Struct(Box::new(Struct::new(
                    "S",
                    vec![("x".into(), LowLevelType::Signed)],
                )))),
                TO: Box::new(LowLevelType::Signed),
                offsets: vec![InteriorOffset::Field("x".into())],
            })),
        );
        let hl = HLOperation::new(OpKind::Bool, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    #[should_panic(expected = "non-constant field-name")]
    fn consider_someptr_getattr_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let mut v_ptr = Variable::named("p");
        ann.setbinding(&mut v_ptr, SomeValue::Ptr(s_ptr));
        let mut v_attr = Variable::named("name");
        ann.setbinding(
            &mut v_attr,
            SomeValue::String(super::super::model::SomeString::new(false, false)),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_ptr), Hlvalue::Variable(v_attr)],
        );

        let _ = hl.consider(&ann);
    }

    #[test]
    #[should_panic(expected = "non-constant field-name")]
    fn consider_someptr_setattr_uses_ptr_override() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, Ptr, PtrTarget,
        };

        let ann = mk_ann();
        let s_ptr = super::super::model::SomePtr::new(Ptr {
            TO: PtrTarget::Func(FuncType {
                args: vec![],
                result: LowLevelType::Void,
            }),
        });
        let mut v_ptr = Variable::named("p");
        ann.setbinding(&mut v_ptr, SomeValue::Ptr(s_ptr));
        let mut v_attr = Variable::named("name");
        ann.setbinding(
            &mut v_attr,
            SomeValue::String(super::super::model::SomeString::new(false, false)),
        );
        let mut v_value = Variable::named("value");
        ann.setbinding(
            &mut v_value,
            SomeValue::Integer(SomeInteger::new(false, false)),
        );
        let hl = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_ptr),
                Hlvalue::Variable(v_attr),
                Hlvalue::Variable(v_value),
            ],
        );

        let _ = hl.consider(&ann);
    }

    #[test]
    fn consider_somebool_invert_returns_someinteger() {
        let (hl, ann) = hl1(OpKind::Invert, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_hash_reifies_annotator_error() {
        let (hl, ann) = hl1(OpKind::Hash, SomeValue::object());
        let err = hl
            .consider(&ann)
            .expect_err("hash() must return AnnotatorError");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("cannot use hash() in RPython")
        );
    }

    #[test]
    fn consider_sometuple_getslice_constant_bounds() {
        // unaryop.py SomeTuple.getslice — constant start/stop slice the
        // items list. A huge `r_uint` bound clamps to len, never wraps
        // negative.
        let ann = mk_ann();
        let mut v0 = Variable::named("t");
        let mut v_start = Variable::named("start");
        let mut v_stop = Variable::named("stop");
        ann.setbinding(
            &mut v0,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::String(SomeString::new(false, false)),
                SomeValue::Bool(SomeBool::new()),
            ])),
        );
        let mut start = SomeInteger::default();
        start.base.const_box = Some(Constant::new(ConstValue::UInt(1)));
        let mut stop = SomeInteger::default();
        stop.base.const_box = Some(Constant::new(ConstValue::UInt(u64::MAX)));
        ann.setbinding(&mut v_start, SomeValue::Integer(start));
        ann.setbinding(&mut v_stop, SomeValue::Integer(stop));
        let hl = HLOperation::new(
            OpKind::GetSlice,
            vec![
                Hlvalue::Variable(v0),
                Hlvalue::Variable(v_start),
                Hlvalue::Variable(v_stop),
            ],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Tuple(t) => assert_eq!(t.items.len(), 2),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_sometuple_len_is_constant() {
        // unaryop.py — len returns immutablevalue(len(items)).
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Float(SomeFloat::new()),
                SomeValue::Bool(SomeBool::new()),
            ])),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(3));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_sometuple_iter_returns_someiterator() {
        let (hl, ann) = hl1(OpKind::Iter, SomeValue::Tuple(SomeTuple::new(vec![])));
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_str_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Str, SomeValue::object());
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_hex_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Hex, SomeValue::object());
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_pos_returns_impossible() {
        // unaryop.py — default pos returns s_ImpossibleValue.
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::object());
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    fn mk_list_of(item: SomeValue) -> super::super::model::SomeList {
        let ann = mk_ann();
        let listdef = super::super::listdef::ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            item,
            false,
            false,
        );
        super::super::model::SomeList::new(listdef)
    }

    #[test]
    fn consider_somelist_contains_returns_bool() {
        let ann = mk_ann();
        let mut v0 = Variable::named("lst");
        let mut v1 = Variable::named("x");
        ann.setbinding(
            &mut v0,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somelist_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    /// A fixed array of synthetic tuples yields the tuple classdef from
    /// `next`, so `.1` (`__pos_1`) reads the constructor's field.
    #[test]
    fn array_of_synthetic_tuples_next_element_keeps_tuple_classdef() {
        use super::super::super::flowspace::model::HostObject;
        use super::super::classdesc::{ClassDef, ClassDesc};
        use super::super::model::SomeInstance;

        let ann = mk_ann();
        let bk = Rc::clone(&ann.bookkeeper);
        let tuple_name = "Tuple<*mut PyObject,str>";
        let tuple_host = HostObject::new_class(tuple_name, vec![]);
        let tuple_desc = ClassDesc::new(&bk, tuple_host, Some(tuple_name.to_string()), None, None)
            .expect("tuple ClassDesc");
        let tuple_cd = ClassDef::new(&bk, &tuple_desc);
        ClassDef::generalize_attr(
            &tuple_cd,
            "__pos_1",
            Some(SomeValue::String(SomeString::new(false, true))),
        )
        .expect("tuple __pos_1");
        let elem = SomeValue::Instance(SomeInstance::new(
            Some(tuple_cd),
            false,
            std::collections::BTreeMap::new(),
        ));

        let array_name = "Array<Tuple<*mut PyObject,str>;2>";
        let array_host = HostObject::new_class(array_name, vec![]);
        let array_desc = ClassDesc::new(&bk, array_host, Some(array_name.to_string()), None, None)
            .expect("array ClassDesc");
        let array_cd = ClassDef::new(&bk, &array_desc);
        ClassDef::generalize_attr(&array_cd, "__pos_0", Some(elem.clone())).expect("array __pos_0");
        ClassDef::generalize_attr(&array_cd, "__pos_1", Some(elem)).expect("array __pos_1");
        let array = SomeValue::Instance(SomeInstance::new(
            Some(array_cd),
            false,
            std::collections::BTreeMap::new(),
        ));

        let mut arr = Variable::named("arr");
        ann.setbinding(&mut arr, array);
        let iter_hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(arr)]);
        assert!(
            iter_hl.transform(&ann).is_none(),
            "shaped array iter must not rewrite to __iter__"
        );
        let s_iter = iter_hl.consider(&ann).unwrap().unwrap();
        let mut it = Variable::named("it");
        ann.setbinding(&mut it, s_iter);
        let next_hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(it)]);
        let s_elem = next_hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Instance(inst) = &s_elem else {
            panic!("element annotation must be the tuple instance, got {s_elem:?}");
        };
        let class_name = inst
            .classdef
            .as_ref()
            .expect("element classdef")
            .borrow()
            .name
            .clone();
        assert_eq!(class_name, tuple_name);

        let mut ev = Variable::named("elem");
        ann.setbinding(&mut ev, s_elem);
        let get = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(ev),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("__pos_1"))),
            ],
        );
        let field = get.consider(&ann).unwrap().unwrap();
        assert!(
            matches!(field, SomeValue::String(_)),
            ".__pos_1 of the element, got {field:?}"
        );
    }

    #[test]
    fn consider_somedict_iter_returns_iterator() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let mut v = Variable::named("d");
        ann.setbinding(&mut v, SomeValue::Dict(s_dict));
        let hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_len_const_returns_const_int() {
        // unaryop.py — len(const "abc") → SomeInteger(const=3).
        let mut s = SomeString::new(false, false);
        s.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("abc")));
        let (hl, ann) = hl1(OpKind::Len, SomeValue::String(s));
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(3));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somechar_len_is_const_1() {
        // unaryop.py — SomeChar.len() → immutablevalue(1).
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::Char(super::super::model::SomeChar::new(false)),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(1));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somechar_ord_returns_nonneg_integer() {
        let (hl, ann) = hl1(
            OpKind::Ord,
            SomeValue::Char(super::super::model::SomeChar::new(false)),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_someiterator_iter_returns_self() {
        // unaryop.py — iter on iterator returns self.
        let ann = mk_ann();
        let it = SomeIterator::new(SomeValue::Tuple(SomeTuple::new(vec![])), vec![]);
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_on_tuple_unions_items() {
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
            vec![],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_enumerate_returns_tuple_int_item() {
        // upstream unaryop.py:822-824 — enumerate variant yields
        // SomeTuple((SomeInteger(nonneg=True), getanyitem(container))).
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
            vec!["enumerate".to_string()],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Tuple(t) = r else {
            panic!("expected SomeTuple");
        };
        assert_eq!(t.items.len(), 2);
        assert!(matches!(t.items[0], SomeValue::Integer(_)));
        assert!(matches!(t.items[1], SomeValue::Integer(_)));
    }

    #[test]
    fn consider_someiterator_next_reversed_acts_as_default() {
        // upstream unaryop.py — `("reversed",)` collapses to ().
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
            vec!["reversed".to_string()],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Integer(_)));
    }

    #[test]
    fn consider_someiterator_next_on_string_returns_char() {
        use super::super::model::SomeString;
        let ann = mk_ann();
        let it = SomeIterator::new(SomeValue::String(SomeString::new(false, true)), vec![]);
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Char(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_on_unicode_returns_codepoint() {
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::UnicodeString(SomeUnicodeString::new(false, true)),
            vec![],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::UnicodeCodePoint(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somenone_getattr_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::GetAttr,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somepbc_class_name_returns_somestring() {
        let ann = mk_ann();
        let class_host =
            super::super::super::flowspace::model::HostObject::new_class("Abc", vec![]);
        let desc = ann.bookkeeper.getdesc(&class_host).expect("class desc");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(
            &mut v_cls,
            SomeValue::PBC(super::super::model::SomePBC::new([desc], false)),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v_cls),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("__name__"))),
            ],
        );
        let r = hl.consider(&ann).expect("class __name__ getattr").unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somepbc_single_desc_routes_through_s_read_attribute() {
        // upstream pbc_getattr single-desc fast path: `if len(descs) == 1:
        // return first.s_read_attribute(attr)`. Construct a ClassDesc PBC
        // whose class has a const attribute and read it through PBC.getattr.
        let ann = mk_ann();
        let class_host =
            super::super::super::flowspace::model::HostObject::new_class("Box", vec![]);
        class_host.class_set("N", ConstValue::Int(42));
        let desc = ann.bookkeeper.getdesc(&class_host).expect("class desc");
        let mut v_cls = Variable::named("cls");
        ann.setbinding(
            &mut v_cls,
            SomeValue::PBC(super::super::model::SomePBC::new([desc], false)),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v_cls),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("N"))),
            ],
        );
        let r = hl
            .consider(&ann)
            .expect("class const attr getattr")
            .unwrap();
        assert!(
            matches!(r.const_(), Some(ConstValue::Int(42))),
            "got {:?}",
            r
        );
    }

    #[test]
    fn consider_somenone_len_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::String(SomeString::new(false, false)),
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_contains_string_nul_sets_false_branch_nonnul_refinement() {
        let ann = mk_ann();
        let mut v_string = Variable::named("s");
        ann.setbinding(
            &mut v_string,
            SomeValue::String(SomeString::new(false, false)),
        );
        let mut nul_char = super::super::model::SomeChar::new(false);
        nul_char.inner.base.const_box = Some(Constant::new(ConstValue::byte_str("\0")));
        let mut v_char = Variable::named("c");
        ann.setbinding(&mut v_char, SomeValue::Char(nul_char));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![
                Hlvalue::Variable(v_string.clone()),
                Hlvalue::Variable(v_char),
            ],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        let narrowed = ktd
            .get(&ExitCaseKey::Bool(false))
            .and_then(|inner| inner.get(&Rc::new(v_string)))
            .expect("false branch must refine the string");
        match narrowed {
            SomeValue::String(s) => assert!(s.inner.no_nul),
            other => panic!("expected SomeString refinement, got {other:?}"),
        }
    }

    #[test]
    fn consider_contains_unicode_nul_sets_false_branch_nonnul_refinement() {
        let ann = mk_ann();
        let mut v_string = Variable::named("u");
        ann.setbinding(
            &mut v_string,
            SomeValue::UnicodeString(SomeUnicodeString::new(false, false)),
        );
        let mut nul_cp = super::super::model::SomeUnicodeCodePoint::new(false);
        nul_cp.inner.base.const_box = Some(Constant::new(ConstValue::uni_str("\0")));
        let mut v_char = Variable::named("c");
        ann.setbinding(&mut v_char, SomeValue::UnicodeCodePoint(nul_cp));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![
                Hlvalue::Variable(v_string.clone()),
                Hlvalue::Variable(v_char),
            ],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        let SomeValue::Bool(b) = r else {
            panic!("expected SomeBool");
        };
        let ktd = b.knowntypedata.expect("knowntypedata must be populated");
        let narrowed = ktd
            .get(&ExitCaseKey::Bool(false))
            .and_then(|inner| inner.get(&Rc::new(v_string)))
            .expect("false branch must refine the unicode string");
        match narrowed {
            SomeValue::UnicodeString(s) => assert!(s.inner.no_nul),
            other => panic!("expected SomeUnicodeString refinement, got {other:?}"),
        }
    }

    #[test]
    fn consider_somedict_contains_position_required() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(3, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let mut v0 = Variable::named("d");
        let mut v1 = Variable::named("k");
        ann.setbinding(&mut v0, SomeValue::Dict(s_dict));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn list_method_extend_uses_iter_next_for_non_list_iterable() {
        let ann = mk_ann();
        let s_list = mk_list_of(SomeValue::Integer(SomeInteger::default()));
        let s_iterable = SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
            SomeInteger::default(),
        )]));
        let result = list_method_extend(&ann, &s_list, &s_iterable);
        assert!(result.is_none(), "method_extend is a void analyser → None");
        assert!(matches!(
            s_list.listdef.read_item(None),
            SomeValue::Integer(_)
        ));
    }

    #[test]
    fn builtin_method_call_binds_keywords_like_python_call() {
        let ann = mk_ann();
        let method = SomeBuiltinMethod::new(
            "str_method_find",
            SomeValue::String(SomeString::new(false, false)),
            "find",
        );
        let mut kwds = std::collections::HashMap::new();
        kwds.insert("start".to_string(), Some(s_const_int(1)));
        kwds.insert("end".to_string(), Some(s_const_int(3)));
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::String(SomeString::new(false, false)))],
            Some(kwds),
            None,
        );
        let result = call_builtin_method(&ann, &method, &args).expect("builtin call must bind");
        assert!(matches!(result, Some(SomeValue::Integer(_))));
    }

    #[test]
    fn find_method_exposes_dict_items_and_string_split_surface() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = SomeValue::Dict(super::super::model::SomeDict::new(dictdef));
        let s_string = SomeValue::String(SomeString::new(false, false));

        let SomeValue::BuiltinMethod(dict_items) = s_dict.find_method("items").unwrap() else {
            panic!("dict.items must be recognized");
        };
        assert_eq!(dict_items.analyser_name, "dict_method_items");

        let SomeValue::BuiltinMethod(str_split) = s_string.find_method("split").unwrap() else {
            panic!("str.split must be recognized");
        };
        assert_eq!(str_split.analyser_name, "str_method_split");
    }

    #[test]
    fn find_method_exposes_stringbuilder_append_build_getlength() {
        let ann = mk_ann();
        let s_builder = SomeValue::StringBuilder(super::super::model::SomeStringBuilder::new());

        for (method_name, analyser) in [
            ("append", "stringbuilder_method_append"),
            ("build", "stringbuilder_method_build"),
            ("getlength", "stringbuilder_method_getlength"),
        ] {
            let SomeValue::BuiltinMethod(bound) = s_builder.find_method(method_name).unwrap()
            else {
                panic!("StringBuilder.{method_name} must be recognized");
            };
            assert_eq!(bound.analyser_name, analyser);
        }
        assert!(s_builder.find_method("frobnicate").is_none());

        // build() → SomeString.
        let build =
            SomeBuiltinMethod::new("stringbuilder_method_build", s_builder.clone(), "build");
        let no_args = super::super::argument::ArgumentsForTranslation::new(vec![], None, None);
        assert!(matches!(
            call_builtin_method(&ann, &build, &no_args).expect("build call"),
            Some(SomeValue::String(_))
        ));

        // getlength() → SomeInteger.
        let getlength = SomeBuiltinMethod::new(
            "stringbuilder_method_getlength",
            s_builder.clone(),
            "getlength",
        );
        let no_args2 = super::super::argument::ArgumentsForTranslation::new(vec![], None, None);
        assert!(matches!(
            call_builtin_method(&ann, &getlength, &no_args2).expect("getlength call"),
            Some(SomeValue::Integer(_))
        ));

        // append(str) → void (None).
        let append =
            SomeBuiltinMethod::new("stringbuilder_method_append", s_builder.clone(), "append");
        let str_arg = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::String(SomeString::new(false, false)))],
            None,
            None,
        );
        assert!(
            call_builtin_method(&ann, &append, &str_arg)
                .expect("append call")
                .is_none(),
            "append is a void analyser → None"
        );
    }

    #[test]
    fn rust_vec_ops_annotate_from_the_item_kind() {
        use crate::flowspace::model::{Constant, Hlvalue};
        use majit_ir::rvec::VecItemKind;
        let ann = mk_ann();
        for kind in VecItemKind::ALL {
            let letter = Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                kind.kind_char().to_string(),
            )));
            let hl = HLOperation::new(OpKind::NewRustVec, vec![letter]);
            let Some(SomeValue::RustVec(s_vec)) = hl.consider(&ann).unwrap() else {
                panic!("newrustvec must type as SomeRustVec");
            };
            assert_eq!(s_vec.item_kind(), Some(kind));
            let s_self = SomeValue::RustVec(s_vec);
            for (name, analyser) in [
                ("append", "rustvec_method_append"),
                ("reverse", "rustvec_method_reverse"),
                ("free", "rustvec_method_free"),
                ("items", "rustvec_method_items"),
                ("extend_from_slice", "rustvec_method_extend_from_slice"),
                ("extend", "rustvec_method_extend"),
            ] {
                let SomeValue::BuiltinMethod(bound) = s_self.find_method(name).unwrap() else {
                    panic!("Vec.{name} must be recognized");
                };
                assert_eq!(bound.analyser_name, analyser);
            }
        }
        let bad = HLOperation::new(
            OpKind::NewRustVec,
            vec![Hlvalue::Constant(Constant::new(ConstValue::byte_str("x")))],
        );
        assert!(bad.consider(&ann).is_err());
    }

    #[test]
    fn rust_vec_setattr_len_annotates_as_void_and_rejects_other_fields() {
        use super::super::model::SomeRustVec;
        use majit_ir::rvec::VecItemKind;
        let ann = mk_ann();
        let s_vec = SomeValue::RustVec(SomeRustVec::for_kind(VecItemKind::Int));
        let mut v_vec = Variable::named("v");
        ann.setbinding(&mut v_vec, s_vec.clone());
        let mut v_n = Variable::named("n");
        ann.setbinding(
            &mut v_n,
            SomeValue::Integer(SomeInteger::new_with_knowntype(true, KnownType::Ruint)),
        );
        let hl_len = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_vec.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("len"))),
                Hlvalue::Variable(v_n.clone()),
            ],
        );
        assert!(
            hl_len.consider(&ann).expect("setattr len").is_none(),
            "SomePtr.setattr is void"
        );

        let hl_signed = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_vec.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("len"))),
                Hlvalue::Variable({
                    let mut v = Variable::named("n_signed");
                    ann.setbinding(&mut v, SomeValue::Integer(SomeInteger::new(true, false)));
                    v
                }),
            ],
        );
        assert!(
            hl_signed
                .consider(&ann)
                .expect("setattr len Signed")
                .is_none()
        );

        let hl_cap = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_vec.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("cap"))),
                Hlvalue::Variable(v_n),
            ],
        );
        let err = hl_cap
            .consider(&ann)
            .expect_err("setattr of cap must stay an annotator error");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("no field \"cap\""),
            "got {err:?}"
        );

        let hl_get = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v_vec.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("len"))),
            ],
        );
        let got = hl_get.consider(&ann).expect("getattr len").unwrap();
        assert!(matches!(got, SomeValue::Integer(_)), "got {got:?}");

        let hl_append = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v_vec),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("append"))),
            ],
        );
        let SomeValue::BuiltinMethod(bound) =
            hl_append.consider(&ann).expect("getattr append").unwrap()
        else {
            panic!("Vec.append must still resolve through find_method");
        };
        assert_eq!(bound.analyser_name, "rustvec_method_append");
    }

    #[test]
    fn consider_newstringbuilder_types_as_stringbuilder() {
        // The `newstringbuilder` ctor op — the front-emitted `StringBuilder()`
        // — types its result as `SomeStringBuilder` (rlib/rstring.py
        // ExtRegistryEntry `compute_result_annotation`).
        let ann = mk_ann();
        let hl = HLOperation::new(OpKind::NewStringBuilder, vec![]);
        let r = hl.consider(&ann).unwrap().unwrap();
        assert!(matches!(r, SomeValue::StringBuilder(_)), "got {r:?}");
    }

    #[test]
    fn classdef_less_instance_is_null_bound_method_calls_to_bool() {
        // `p.is_null()` on a pointer-erased (classdef-less) `SomeInstance`
        // lowers as `getattr(p, "is_null")` + `simple_call(bound, …)`
        // (CallTarget::Method).  The bound method's call must yield
        // `SomeBool`, so the callee slot holds a callable rather than a
        // bare bool.
        let ann = mk_ann();
        let instance = SomeValue::Instance(super::super::model::SomeInstance::new(
            None,
            false,
            std::collections::BTreeMap::new(),
        ));
        let method = SomeBuiltinMethod::new("ptr_method_is_null", instance, "is_null");
        let args = super::super::argument::ArgumentsForTranslation::new(vec![], None, None);
        let result =
            call_builtin_method(&ann, &method, &args).expect("is_null bound method must call");
        assert!(matches!(result, Some(SomeValue::Bool(_))));
    }

    #[test]
    fn classdef_less_instance_getattr_is_null_returns_bound_method() {
        // Drive the registered `SomeInstance.getattr` specialization (not
        // a hand-built `SomeBuiltinMethod`) so the classdef-less
        // `is_null` arm at the GetAttr/Instance entry stays pinned: the
        // getattr must return the callable bound method, never a bare
        // `SomeBool` in the simple_call callee slot.
        let ann = mk_ann();
        let mut v = Variable::named("p");
        ann.setbinding(
            &mut v,
            SomeValue::Instance(super::super::model::SomeInstance::new(
                None,
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("is_null"))),
            ],
        );

        let result = hl.consider(&ann).expect("getattr must succeed").unwrap();

        let SomeValue::BuiltinMethod(method) = result else {
            panic!("expected SomeBuiltinMethod, got a non-callable");
        };
        assert_eq!(method.analyser_name, "ptr_method_is_null");
        assert_eq!(method.methodname, "is_null");
        assert!(
            matches!(*method.s_self, SomeValue::Instance(_)),
            "bound method must carry the erased receiver"
        );
    }

    #[test]
    fn classed_instance_getattr_is_null_dispatch() {
        // The ptr-method arm answers a classed receiver (`SomeInstance(cd)`
        // from `cast_pointer`) only while the class hierarchy defines no
        // `is_null` of its own — a real classdict member keeps upstream's
        // `SomeInstance.getattr` classdef dispatch (unaryop.py).
        use super::super::classdesc::{ClassDef, ClassDictEntry};
        use super::super::model::SomeInstance;
        let ann = mk_ann();

        // No `is_null` member → the ptr method answers.
        let bare = ClassDef::new_standalone("pkg.W_Bare", None);
        let mut v = Variable::named("p");
        ann.setbinding(
            &mut v,
            SomeValue::Instance(SomeInstance::new(
                Some(bare),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("is_null"))),
            ],
        );
        let result = hl.consider(&ann).expect("getattr must succeed").unwrap();
        let SomeValue::BuiltinMethod(method) = result else {
            panic!("expected the ptr bound method on a member-less class");
        };
        assert_eq!(method.analyser_name, "ptr_method_is_null");

        // A real `is_null` classdict member shadows the ptr method and
        // resolves through the classdef.  Built against the annotator's
        // live bookkeeper (`new_standalone` drops its own, and the
        // classdef path needs the backlink for `s_get_value`).
        use super::super::classdesc::ClassDesc;
        let pyobj = super::super::super::flowspace::model::HostObject::new_class(
            "pkg.W_Shadow",
            Vec::new(),
        );
        let classdesc = std::rc::Rc::new(std::cell::RefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            pyobj,
            "pkg.W_Shadow".to_string(),
        )));
        let shadowing = ClassDef::new(&ann.bookkeeper, &classdesc);
        classdesc.borrow_mut().classdict.insert(
            "is_null".to_string(),
            ClassDictEntry::constant(ConstValue::Int(7)),
        );
        let mut v2 = Variable::named("q");
        ann.setbinding(
            &mut v2,
            SomeValue::Instance(SomeInstance::new(
                Some(shadowing),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let hl2 = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v2),
                Hlvalue::Constant(Constant::new(ConstValue::byte_str("is_null"))),
            ],
        );
        let result2 = hl2.consider(&ann).expect("getattr must succeed").unwrap();
        assert!(
            !matches!(
                &result2,
                SomeValue::BuiltinMethod(m) if m.analyser_name == "ptr_method_is_null"
            ),
            "a class member named is_null must win over the ptr method, got {result2:?}"
        );
    }

    #[test]
    fn str_method_upper_preserves_unicode_family() {
        let ann = mk_ann();
        let unicode = SomeValue::UnicodeString(SomeUnicodeString::new(false, false));
        let codepoint =
            SomeValue::UnicodeCodePoint(super::super::model::SomeUnicodeCodePoint::new(false));

        assert!(matches!(
            str_method_upper(&ann, &unicode),
            SomeValue::UnicodeString(_)
        ));
        assert!(matches!(
            str_method_upper(&ann, &codepoint),
            SomeValue::UnicodeString(_)
        ));
    }

    #[test]
    fn str_method_lower_preserves_unicode_family() {
        let ann = mk_ann();
        let unicode = SomeValue::UnicodeString(SomeUnicodeString::new(false, false));
        let codepoint =
            SomeValue::UnicodeCodePoint(super::super::model::SomeUnicodeCodePoint::new(false));

        assert!(matches!(
            str_method_lower(&ann, &unicode),
            SomeValue::UnicodeString(_)
        ));
        assert!(matches!(
            str_method_lower(&ann, &codepoint),
            SomeValue::UnicodeString(_)
        ));
    }

    #[test]
    fn dict_method_items_returns_list_of_pairs() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(7, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let result = dict_method_items(&ann, &s_dict);
        let SomeValue::List(list) = result else {
            panic!("expected SomeList");
        };
        assert!(matches!(list.listdef.read_item(None), SomeValue::Tuple(_)));
    }

    #[test]
    fn find_method_exposes_dict_hash_and_move_to_end_surface() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = SomeValue::Dict(super::super::model::SomeDict::new(dictdef));

        let SomeValue::BuiltinMethod(getitem) = s_dict.find_method("getitem_with_hash").unwrap()
        else {
            panic!("dict.getitem_with_hash must be recognized");
        };
        assert_eq!(getitem.analyser_name, "dict_method_getitem_with_hash");

        let SomeValue::BuiltinMethod(move_to_end) = s_dict.find_method("move_to_end").unwrap()
        else {
            panic!("dict.move_to_end must be recognized");
        };
        assert_eq!(move_to_end.analyser_name, "dict_method_move_to_end");
    }

    #[test]
    fn dict_method_getitem_with_hash_reads_value_type() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(9, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let result = dict_method_getitem_with_hash(
            &ann,
            &s_dict,
            &SomeValue::Integer(SomeInteger::default()),
            &SomeValue::Integer(SomeInteger::default()),
        );
        assert!(matches!(result, SomeValue::String(_)));
    }

    #[test]
    fn dict_method_insert_on_instance_values_admits_none_and_generalizes_value() {
        // Old lowering annotated get(key) (dfl = s_None) then
        // setitem(key, value). insert(key, value) must keep that pair:
        // the result admits None (miss returns NULL) and the dict
        // value cell is generalized with the inserted instance.
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;

        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(9, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let s_inst = SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            s_inst.clone(),
            false,
            false,
            false,
        );
        let s_dict = SomeValue::Dict(super::super::model::SomeDict::new(dictdef));

        let SomeValue::BuiltinMethod(insert) = s_dict.find_method("insert").unwrap() else {
            panic!("dict.insert must be recognized");
        };
        assert_eq!(insert.analyser_name, "dict_method_insert");
        let SomeValue::BuiltinMethod(get) = s_dict.find_method("get").unwrap() else {
            panic!("dict.get must be recognized");
        };
        assert_eq!(get.analyser_name, "dict_method_get");
        let SomeValue::BuiltinMethod(setdefault) = s_dict.find_method("setdefault").unwrap() else {
            panic!("dict.setdefault must be recognized");
        };
        assert_eq!(setdefault.analyser_name, "dict_method_get");

        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![
                Some(SomeValue::Integer(SomeInteger::default())),
                Some(s_inst),
            ],
            None,
            None,
        );
        let result = call_builtin_method(&ann, &insert, &args).expect("insert call must bind");
        let Some(SomeValue::Instance(result_inst)) = result else {
            panic!("insert must return an instance annotation, got {result:?}");
        };
        assert!(
            result_inst.can_be_none,
            "insert result must admit None (miss returns NULL)"
        );

        let SomeValue::Dict(s_self) = &s_dict else {
            panic!("receiver stays a dict");
        };
        let stored = s_self.dictdef.read_value(None);
        let SomeValue::Instance(stored_inst) = &stored else {
            panic!("dict value must stay an instance, got {stored:?}");
        };
        assert!(
            stored_inst.can_be_none,
            "dict value must be generalized with None from the miss path"
        );
        assert!(
            stored_inst
                .classdef
                .as_ref()
                .is_some_and(|cd| Rc::ptr_eq(cd, &classdef)),
            "dict value must stay generalized with the inserted instance class"
        );
    }

    #[test]
    fn dict_method_insert_rejects_integer_value_that_cannot_noneify() {
        // `SomeInteger.noneify` raises `UnionError` (`model.py`). insert's
        // miss path is NULL, so an integer-valued dict cannot be annotated.
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(11, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
            false,
        );
        let s_dict = SomeValue::Dict(super::super::model::SomeDict::new(dictdef));
        let SomeValue::BuiltinMethod(insert) = s_dict.find_method("insert").unwrap() else {
            panic!("dict.insert must be recognized");
        };
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![
                Some(SomeValue::Integer(SomeInteger::default())),
                Some(SomeValue::Integer(SomeInteger::default())),
            ],
            None,
            None,
        );
        let err = call_builtin_method(&ann, &insert, &args)
            .expect_err("integer-valued insert must fail annotation");
        let msg = err.msg.unwrap_or_default();
        assert!(
            msg.contains("cannot represent None"),
            "expected noneify miss diagnostic, got {msg}"
        );
    }

    #[test]
    fn builtin_method_call_binds_hash_method_keywords() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let method = SomeBuiltinMethod::new(
            "dict_method_contains_with_hash",
            SomeValue::Dict(super::super::model::SomeDict::new(dictdef)),
            "contains_with_hash",
        );
        let mut kwds = std::collections::HashMap::new();
        kwds.insert("hash".to_string(), Some(s_const_int(11)));
        let args = super::super::argument::ArgumentsForTranslation::new(
            vec![Some(SomeValue::Integer(SomeInteger::default()))],
            Some(kwds),
            None,
        );
        let result = call_builtin_method(&ann, &method, &args).expect("builtin call must bind");
        assert!(matches!(result, Some(SomeValue::Bool(_))));
    }
}
