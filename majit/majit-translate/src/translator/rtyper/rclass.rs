//! RPython `rpython/rtyper/rclass.py`.
//!
//! Current port scope:
//!
//! * Module-level type constants `OBJECT_VTABLE`, `CLASSTYPE`, `OBJECT`,
//!   `OBJECTPTR`, `NONGCOBJECT`, `NONGCOBJECTPTR` (rclass.py).
//!   Backed by `LazyLock<LowLevelType>` to preserve the upstream
//!   module-level singleton semantics (module import = first Lazy deref).
//! * `RootClassRepr` (rclass.py), `ClassRepr` (rclass.py),
//!   `InstanceRepr` (rclass.py) — `__init__` for all three and
//!   the `_setup_repr` vtable-super / object_type-super chains landed,
//!   including readonly/non-readonly attr iteration, `prepare_method`,
//!   `extra_access_sets`, and field reordering by `attr_reverse_size`.
//! * `getclassrepr` / `getinstancerepr` / `buildinstancerepr`
//!   (rclass.py) — both `classdef is None` and
//!   `classdef != None` arms dispatch. `getclassrepr` caches through
//!   upstream's `classdef.repr` slot, and `getinstancerepr` reads
//!   `_alloc_flavor_` through `classdesc.get_param(...)`.
//!
//! Deferred:
//!
//! * `_check_for_immutable_hints` (rclass.py),
//!   `special_memory_pressure` + `mutate_*` quasi-immutable fields
//!   (rclass.py). Phase R2-D, gated on `classdesc.get_param`
//!   / bookkeeper `memory_pressure_types`.
//!
//! Phase R3 landed (`ClassRepr.init_vtable` at [`ClassRepr::init_vtable`]
//! line 838, `fill_vtable_root` at [`ClassRepr::fill_vtable_root`] line
//! 891, `setup_vtable` at [`ClassRepr::setup_vtable`] line 678,
//! `getvtable` at line 1023, `getruntime` at line 1042) — full
//! `lltype.malloc(immortal=True)` / `attachRuntimeTypeInfo` /
//! `RuntimeTypeInfo` wiring is in place; the `RootClassRepr.init_vtable`
//! override (rclass.py) is at line 1682.

#![allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    private_interfaces
)]

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};

/// RPython `flags={}` keyword argument threaded through
/// `InstanceRepr.{getfield,setfield}` (rclass.py / :1002) and the
/// `hook_access_field` / `hook_setfield` virtualizable hooks
/// (rclass.py). Default empty maps to upstream's `{}` literal;
/// `_jit_virtualizable_2_` instance reprs read keys like `'access_directly'`
/// and `'fresh_virtualizable'` from this dict.
pub(crate) type Flags = HashMap<String, ConstValue>;
use std::rc::{Rc, Weak};
use std::sync::{Arc, LazyLock};

thread_local! {
    // rclass.py:InstanceRepr._initialize_data_flattenrec is a class-owned
    // FlattenRecursion; upstream tool/flattenrec.py derives from thread._local.
    // Share one queue across reprs on the translation thread, not per instance.
    static INITIALIZE_DATA_FLATTENREC: crate::tool::flattenrec::FlattenRecursion<TyperError> =
        crate::tool::flattenrec::FlattenRecursion::new();
}

use crate::annotator::classdesc::ClassDef;
use crate::annotator::description::{ClassDefKey, DescEntry};
use crate::annotator::model::{DescKind, SomePBC, SomeValue};
use crate::codewriter::type_state::ConcreteType;
use crate::flowspace::model::{ConstValue, Constant, Hlvalue, HostObject, Variable};
use crate::model::{BlockId, FunctionGraph, OpKind, SpaceOperation};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    self, _ptr, ForwardReference, LowLevelType, Ptr, PtrTarget, RUNTIME_TYPE_INFO, Struct,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{DescOrConst, RTypeResult, Repr, ReprState, mangle};
use crate::translator::rtyper::rtyper::{
    GenopResult, HighLevelOp, LowLevelFunction, LowLevelOpList, RPythonTyper,
};

// VtableMethodPtr helper (carried over from the pre-R1 rclass.rs scaffold).

/// Insert a `VtableMethodPtr` op at `(block_id, op_index)` and return the
/// produced funcptr Variable. Writes the funcptr's `concretetype` cell to
/// `Signed` so downstream passes (`build_value_kinds`, regalloc, flatten)
/// see the funcptr as integer kind — matching `int_guard_value(op.args[0])`
/// in `jtransform.py:546`.
///
/// RPython equivalent: `ClassRepr.getclsfield(vcls, attr, llops)`
/// (`rclass.py`), which appends `cast_pointer + getfield(vtable,
/// mangled_name)` to `llops`. The full `getclsfield` is ported on
/// [`ClassRepr`] and on the rooted [`RootClassRepr`]; this
/// freestanding helper stays as a
/// TODO: bridge for the pyre IR representation of
/// vtable method slots (see the `OpKind::VtableMethodPtr` comment
/// block in `model.rs`).
pub(crate) fn class_get_method_ptr(
    graph: &mut FunctionGraph,
    block_id: BlockId,
    op_index: usize,
    receiver: Variable,
    trait_root: String,
    method_name: String,
) -> Variable {
    let funcptr_var = graph.alloc_value_var();
    let op = SpaceOperation {
        result: Some(funcptr_var.clone()),
        kind: OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        },
    };
    graph.blocks[block_id.0].operations.insert(op_index, op);
    // RPython parity: every Variable carries `concretetype` inline
    // (`rtyper.py setconcretetype v.concretetype = ...`).  Write through the
    // backing cell so downstream consumers see Signed.
    FunctionGraph::set_concretetype_of_inline(&funcptr_var, ConcreteType::Signed);
    funcptr_var
}

// rclass.py — FieldListAccessor / ImmutableRanking / IR_* constants.

/// RPython `class ImmutableRanking` (`rclass.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImmutableRanking {
    pub name: &'static str,
    pub is_immutable: bool,
}

/// RPython `IR_MUTABLE = ImmutableRanking('mutable', False)` (rclass.py).
pub const IR_MUTABLE: ImmutableRanking = ImmutableRanking {
    name: "mutable",
    is_immutable: false,
};

/// RPython `IR_IMMUTABLE = ImmutableRanking('immutable', True)` (rclass.py).
pub const IR_IMMUTABLE: ImmutableRanking = ImmutableRanking {
    name: "immutable",
    is_immutable: true,
};

/// RPython `class ImmutableConflictError(Exception)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImmutableConflictError {
    pub message: String,
}

/// RPython `IR_IMMUTABLE_ARRAY = ImmutableRanking('immutable_array', True)`.
pub const IR_IMMUTABLE_ARRAY: ImmutableRanking = ImmutableRanking {
    name: "immutable_array",
    is_immutable: true,
};

/// RPython `IR_QUASIIMMUTABLE = ImmutableRanking('quasiimmutable', False)`.
pub const IR_QUASIIMMUTABLE: ImmutableRanking = ImmutableRanking {
    name: "quasiimmutable",
    is_immutable: false,
};

/// RPython `IR_QUASIIMMUTABLE_ARRAY = ImmutableRanking('quasiimmutable_array', False)`.
pub const IR_QUASIIMMUTABLE_ARRAY: ImmutableRanking = ImmutableRanking {
    name: "quasiimmutable_array",
    is_immutable: false,
};

/// RPython `class FieldListAccessor(object)` (`rclass.py`).
#[derive(Clone, Debug, Default)]
pub struct FieldListAccessor {
    pub TYPE: Option<LowLevelType>,
    pub fields: HashMap<String, ImmutableRanking>,
}

impl FieldListAccessor {
    /// RPython `FieldListAccessor.initialize(self, TYPE, fields)`.
    pub fn initialize(&mut self, TYPE: LowLevelType, fields: HashMap<String, ImmutableRanking>) {
        self.TYPE = Some(TYPE);
        self.fields = fields;
    }

    /// RPython `FieldListAccessor.all_immutable_fields(self)`.
    pub fn all_immutable_fields(&self) -> HashSet<String> {
        self.fields
            .iter()
            .filter_map(|(key, value)| {
                if *value == IR_IMMUTABLE || *value == IR_IMMUTABLE_ARRAY {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// RPython `class MissingRTypeAttribute(TyperError)` (`rclass.py`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingRTypeAttribute {
    pub attr: String,
}

// rclass.py — OBJECT_VTABLE / OBJECT / NONGCOBJECT module constants.

/// RPython `alloc_array_name(name)` (`rclass.py`).
pub fn alloc_array_name(name: &str) -> Result<_ptr, String> {
    crate::translator::rtyper::lltypesystem::rstr::alloc_array_name(name)
}

/// Internal aggregate that materialises the four interdependent
/// module-level types (`OBJECT_VTABLE` / `CLASSTYPE` / `OBJECT` /
/// `OBJECTPTR`) in a single `LazyLock` body, mirroring upstream's
/// mutable-ForwardReference + post-hoc `become()` ordering
/// (rclass.py).
///
/// Cycle topology:
///
/// ```text
/// OBJECT_VTABLE.instantiate  →  Ptr(FuncType([], OBJECTPTR))
///                                                   ↓
///                                                 OBJECTPTR
///                                                   ↓
///                                                 OBJECT
///                                                   ↓
///                                          OBJECT.typeptr = CLASSTYPE
///                                                   ↓
///                                              CLASSTYPE = Ptr(OBJECT_VTABLE)
///                                                   ↺
/// ```
///
/// Resolved by:
/// 1. Mint an unresolved `vtable_fwd: ForwardReference`.
/// 2. Build `CLASSTYPE = Ptr(ForwardReference(vtable_fwd.clone()))`
///    — clones share `Arc<Mutex<Option<...>>>` so any later
///    `become()` propagates.
/// 3. Build `OBJECT` and `OBJECTPTR` referencing CLASSTYPE.
/// 4. Build the full `object_vtable` Struct (with all 5 fields
///    including `instantiate: Ptr(FuncType([], OBJECTPTR))`).
/// 5. `vtable_fwd.become(struct_value)` — every clone (including
///    the one embedded in CLASSTYPE) sees the resolution via the
///    shared `Arc`.
struct ObjectFamilyTypes {
    object_vtable: LowLevelType,
    classtype: LowLevelType,
    object: LowLevelType,
    objectptr: LowLevelType,
}

static OBJECT_FAMILY: LazyLock<ObjectFamilyTypes> = LazyLock::new(|| {
    use crate::translator::rtyper::lltypesystem::lltype::FuncType;

    // Step 1 — mint the unresolved ForwardReference.
    let vtable_fwd = ForwardReference::new();

    // Step 2 — CLASSTYPE = Ptr(OBJECT_VTABLE) via a Ptr wrapping a
    // clone of `vtable_fwd`. Clones share the resolved state via
    // `Arc<Mutex<_>>`, so step 5 propagates here.
    let classtype = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(vtable_fwd.clone()),
    }));

    // Step 3a — OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
    // rtti=True). Same body as before, but built from the local
    // `classtype` rather than the static singleton.
    let object = LowLevelType::Struct(Box::new(Struct::gc_rtti_with_hints(
        "object",
        vec![("typeptr".into(), classtype.clone())],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("shouldntbenull".into(), ConstValue::Bool(true)),
            ("typeptr".into(), ConstValue::Bool(true)),
        ],
    )));

    // Step 3b — OBJECTPTR = Ptr(OBJECT).
    let LowLevelType::Struct(object_body) = object.clone() else {
        unreachable!("OBJECT must be a Struct");
    };
    let objectptr = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*object_body),
    }));

    // Step 4 — build the full object_vtable Struct, including the
    // `instantiate: Ptr(FuncType([], OBJECTPTR))` field that closes
    // the cycle.
    let rtti_ptr_type = LowLevelType::Ptr(Box::new(
        Ptr::from_container_type(RUNTIME_TYPE_INFO.clone()).expect(
            "Ptr(RuntimeTypeInfo) must be constructible from the RUNTIME_TYPE_INFO singleton",
        ),
    ));
    let name_ptr_type = crate::translator::rtyper::lltypesystem::rstr::STRPTR.clone();
    // upstream rclass.py — `Ptr(FuncType([], OBJECTPTR))`. The
    // funcptr has zero args and returns OBJECTPTR.
    let instantiate_funcptr_type = LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Func(FuncType {
            args: Vec::new(),
            result: objectptr.clone(),
        }),
    }));
    let body = Struct::with_hints(
        "object_vtable",
        vec![
            ("subclassrange_min".into(), LowLevelType::Signed),
            ("subclassrange_max".into(), LowLevelType::Signed),
            ("rtti".into(), rtti_ptr_type),
            ("name".into(), name_ptr_type),
            ("instantiate".into(), instantiate_funcptr_type),
        ],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("static_immutable".into(), ConstValue::Bool(true)),
        ],
    );

    // Step 5 — resolve.
    vtable_fwd
        .r#become(LowLevelType::Struct(Box::new(body)))
        .expect("OBJECT_VTABLE.become should succeed");
    let object_vtable = LowLevelType::ForwardReference(Box::new(vtable_fwd));

    ObjectFamilyTypes {
        object_vtable,
        classtype,
        object,
        objectptr,
    }
});

/// RPython module-level `OBJECT_VTABLE = lltype.ForwardReference()` resolved
/// via `.become(Struct('object_vtable', ...))` (rclass.py, 167-174).
///
/// All 5 head fields land — `subclassrange_min`/`max`, `rtti`, `name`,
/// and `instantiate: Ptr(FuncType([], OBJECTPTR))`. The instantiate
/// funcptr field closes the `OBJECT_VTABLE → OBJECTPTR → OBJECT →
/// CLASSTYPE → OBJECT_VTABLE` cycle, broken by `OBJECT_FAMILY`'s
/// LazyLock body which builds an unresolved ForwardReference first,
/// constructs the four types in dependency order, then `become()`s
/// the ForwardReference to the full vtable Struct.
pub static OBJECT_VTABLE: LazyLock<LowLevelType> =
    LazyLock::new(|| OBJECT_FAMILY.object_vtable.clone());

/// RPython `CLASSTYPE = Ptr(OBJECT_VTABLE)` (rclass.py).
pub static CLASSTYPE: LazyLock<LowLevelType> = LazyLock::new(|| OBJECT_FAMILY.classtype.clone());

/// RPython `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
/// hints={...}, rtti=True)` (rclass.py). `rtti=True` funnels
/// through `RttiStruct._install_extras` and mints a
/// `RuntimeTypeInfo` opaque stored on `_runtime_type_info`, so
/// `getRuntimeTypeInfo(OBJECT)` succeeds once R3 consumers
/// (`fill_vtable_root`) land.
pub static OBJECT: LazyLock<LowLevelType> = LazyLock::new(|| OBJECT_FAMILY.object.clone());

/// RPython `OBJECTPTR = Ptr(OBJECT)` (rclass.py).
pub static OBJECTPTR: LazyLock<LowLevelType> = LazyLock::new(|| OBJECT_FAMILY.objectptr.clone());

/// RPython `NONGCOBJECT = Struct('nongcobject', ('typeptr', CLASSTYPE))`
/// (rclass.py).
pub static NONGCOBJECT: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(Struct::new(
        "nongcobject",
        vec![("typeptr".into(), CLASSTYPE.clone())],
    )))
});

/// RPython `NONGCOBJECTPTR = Ptr(NONGCOBJECT)` (rclass.py).
pub static NONGCOBJECTPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let LowLevelType::Struct(body) = NONGCOBJECT.clone() else {
        panic!("NONGCOBJECT must be a Struct");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*body),
    }))
});

/// RPython `Flavor` as stored in `getgcflavor(classdef)` return values /
/// `default_flavor` kwargs. Upstream uses strings `'gc'` / `'raw'` /
/// `'stack'` (rclass.py `LLFLAVOR = {'gc':'gc','raw':'raw','stack':'raw'}`).
/// Rust uses a `Copy + Hash + Eq` enum so the `instance_reprs` cache key
/// `(Option<ClassDefKey>, Flavor)` stays pointer-identity friendly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Flavor {
    Gc,
    Raw,
}

impl Flavor {
    /// RPython `LLFLAVOR[flavor]` (rclass.py). `'stack'` folds to
    /// `'raw'` upstream; Rust surfaces it via [`Flavor::from_str_strict`]
    /// when parity of that arm matters.
    pub fn llflavor(self) -> &'static str {
        match self {
            Flavor::Gc => "gc",
            Flavor::Raw => "raw",
        }
    }

    fn from_alloc_flavor(value: &str) -> Result<Self, TyperError> {
        match value {
            "gc" => Ok(Flavor::Gc),
            "raw" | "stack" => Ok(Flavor::Raw),
            other => Err(TyperError::message(format!(
                "unsupported _alloc_flavor_ {other:?} in rclass.getgcflavor"
            ))),
        }
    }
}

fn hlvalue_byte_str(value: &Hlvalue) -> Option<String> {
    let Hlvalue::Constant(c) = value else {
        return None;
    };
    match &c.value {
        ConstValue::ByteStr(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
        ConstValue::UniStr(text) => Some(text.clone()),
        _ => None,
    }
}

fn const_value_field_names(value: &ConstValue) -> Vec<String> {
    match value {
        ConstValue::List(items) | ConstValue::Tuple(items) => items
            .iter()
            .filter_map(|item| match item {
                ConstValue::ByteStr(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
                ConstValue::UniStr(text) => Some(text.clone()),
                _ => None,
            })
            .collect(),
        ConstValue::ByteStr(bytes) => vec![String::from_utf8_lossy(bytes).into_owned()],
        ConstValue::UniStr(text) => vec![text.clone()],
        _ => Vec::new(),
    }
}

pub(crate) fn const_truthy(value: &ConstValue) -> bool {
    match value {
        ConstValue::None => false,
        ConstValue::Placeholder => false,
        ConstValue::Bool(flag) => *flag,
        ConstValue::Int(value) => *value != 0,
        ConstValue::Int128(value) => *value != 0,
        ConstValue::UInt128(value) => *value != 0,
        ConstValue::Float(bits) => f64::from_bits(*bits) != 0.0,
        ConstValue::ByteStr(text) => !text.is_empty(),
        ConstValue::UniStr(text) => !text.is_empty(),
        ConstValue::Tuple(items) => !items.is_empty(),
        ConstValue::List(items) => !items.is_empty(),
        ConstValue::Dict(items) => !items.is_empty(),
        ConstValue::Atom(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::LLPtr(_)
        | ConstValue::LLAddress(_)
        | ConstValue::AddressOffset(_)
        | ConstValue::InheritanceId { .. }
        | ConstValue::SpecTag(_)
        | ConstValue::HostObject(_)
        | ConstValue::Opaque(_)
        | ConstValue::Repr(_) => true,
    }
}

fn host_is_unboxed_value_subclass(pyobj: &HostObject) -> bool {
    pyobj.mro().is_some_and(|mro| {
        mro.iter().any(|cls| {
            let qualname = cls.qualname();
            qualname == "UnboxedValue" || qualname.ends_with(".UnboxedValue")
        })
    })
}

fn classdesc_get_param(
    classdef: &Rc<RefCell<ClassDef>>,
    name: &str,
    default: ConstValue,
    inherit: bool,
) -> ConstValue {
    let mut cdesc = Some(classdef.borrow().classdesc.clone());
    while let Some(cdesc_rc) = cdesc {
        let (value, basedesc) = {
            let cdesc_ref = cdesc_rc.borrow();
            (cdesc_ref.pyobj.class_get(name), cdesc_ref.basedesc.clone())
        };
        if let Some(value) = value {
            return value;
        }
        if !inherit {
            break;
        }
        cdesc = basedesc;
    }
    default
}

pub(super) fn getgcflavor(classdef: &Rc<RefCell<ClassDef>>) -> Result<Flavor, TyperError> {
    let alloc_flavor =
        classdesc_get_param(classdef, "_alloc_flavor_", ConstValue::byte_str("gc"), true);
    let Some(alloc_flavor) = alloc_flavor.as_text() else {
        return Err(TyperError::message(
            "classdesc.get_param('_alloc_flavor_') must return a string",
        ));
    };
    Flavor::from_alloc_flavor(alloc_flavor)
}

fn lowleveltype_size_hint(lltype: &LowLevelType) -> Option<usize> {
    match lltype {
        LowLevelType::Void => None,
        LowLevelType::Bool | LowLevelType::Char => Some(1),
        LowLevelType::UniChar | LowLevelType::SingleFloat => Some(4),
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::SignedLongLong
        | LowLevelType::UnsignedLongLong
        | LowLevelType::Float
        | LowLevelType::LongFloat
        | LowLevelType::Address
        | LowLevelType::Ptr(_)
        | LowLevelType::InteriorPtr(_) => Some(std::mem::size_of::<usize>()),
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => Some(16),
        LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Func(_)
        | LowLevelType::Opaque(_) => None,
        LowLevelType::ForwardReference(fwd) => {
            fwd.resolved().as_ref().and_then(lowleveltype_size_hint)
        }
    }
}

pub fn attr_reverse_size((_, lltype): &(String, LowLevelType)) -> Option<i64> {
    lowleveltype_size_hint(lltype).map(|size| -(size as i64))
}

fn sort_llfields(llfields: &mut [(String, LowLevelType)]) {
    llfields.sort_by(|left, right| left.0.cmp(&right.0));
    llfields.sort_by_key(attr_reverse_size);
}

fn class_attr_family_key(
    access_set: &Rc<RefCell<crate::annotator::description::ClassAttrFamily>>,
) -> usize {
    Rc::as_ptr(access_set) as usize
}

/// RPython `OBJECT_BY_FLAVOR[LLFLAVOR[gcflavor]]` (rclass.py,
/// consumed at rclass.py). Returns the underlying `LowLevelType` for
/// the root `object_type` of an `InstanceRepr` with `classdef is None`.
pub(crate) fn object_by_flavor(flavor: Flavor) -> LowLevelType {
    match flavor {
        Flavor::Gc => OBJECT.clone(),
        Flavor::Raw => NONGCOBJECT.clone(),
    }
}

fn rclass_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!("rclass.{name} helper surface deferred"))
}

/// RPython `ll_inst_hash(ins)`.
pub fn ll_inst_hash(ins: Option<&_ptr>) -> i64 {
    match ins {
        Some(ptr) if ptr.nonzero() => ptr._hashable_identity() as i64,
        _ => 0,
    }
}

/// `ll_inst_hash` (`rclass.py`): a null instance hashes as `0`, otherwise
/// `lltype.identityhash`.
fn build_ll_inst_hash_helper_graph(
    name: &str,
    arg_ty: &LowLevelType,
) -> Result<crate::flowspace::pygraph::PyGraph, TyperError> {
    use crate::flowspace::model::{
        Block, BlockRefExt, FunctionGraph, GraphFunc, Link, SpaceOperation,
    };
    use crate::translator::rtyper::rtyper::{
        constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
    };

    let arg = variable_with_lltype("ins", arg_ty.clone());
    let is_nonnull = variable_with_lltype("is_nonnull", LowLevelType::Bool);
    let ins1 = variable_with_lltype("ins1", arg_ty.clone());
    let hashed = variable_with_lltype("hashed", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let hashblock = Block::shared(vec![Hlvalue::Variable(ins1.clone())]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "ptr_nonzero",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(is_nonnull.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_nonnull));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(arg)],
            Some(hashblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![constant_with_lltype(
                ConstValue::Int(0),
                LowLevelType::Signed,
            )],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);
    hashblock.borrow_mut().operations.push(SpaceOperation::new(
        "gc_identityhash",
        vec![Hlvalue::Variable(ins1)],
        Hlvalue::Variable(hashed.clone()),
    ));
    hashblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(hashed)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ins".to_string()],
        func,
    ))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
struct MissingMarker;

/// RPython `_missing = object()`.
#[allow(dead_code)]
pub(crate) static _missing: MissingMarker = MissingMarker;

/// RPython `fishllattr(inst, name, default=_missing)`.
pub fn fishllattr(
    inst: &_ptr,
    name: &str,
    default: Option<lltype::LowLevelValue>,
) -> Result<lltype::LowLevelValue, TyperError> {
    let field = format!("inst_{name}");
    let mut p = inst.clone();
    loop {
        if let Ok(value) = p.getattr(&field) {
            return Ok(value);
        }
        match p.getattr("super") {
            Ok(lltype::LowLevelValue::Ptr(super_ptr)) => p = *super_ptr,
            _ => break,
        }
    }
    default.ok_or_else(|| {
        TyperError::message(format!("fishllattr: {:?} has no field {name}", inst._TYPE))
    })
}

/// RPython `ll_cast_to_object(obj)`.
pub fn ll_cast_to_object(obj: &_ptr) -> Result<_ptr, TyperError> {
    let LowLevelType::Ptr(objectptr) = OBJECTPTR.clone() else {
        return Err(TyperError::message("OBJECTPTR must be a Ptr"));
    };
    lltype::cast_pointer(&objectptr, obj).map_err(TyperError::message)
}

/// RPython `ll_type(obj)`.
pub fn ll_type(obj: &_ptr) -> Result<_ptr, TyperError> {
    let object = ll_cast_to_object(obj)?;
    match object.getattr("typeptr").map_err(TyperError::message)? {
        lltype::LowLevelValue::Ptr(typeptr) => Ok(*typeptr),
        other => Err(TyperError::message(format!(
            "ll_type: expected Ptr typeptr, got {other:?}"
        ))),
    }
}

/// RPython `ll_runtime_type_info(obj)`.
pub fn ll_runtime_type_info(obj: &_ptr) -> Result<_ptr, TyperError> {
    let typeptr = ll_type(obj)?;
    match typeptr.getattr("rtti").map_err(TyperError::message)? {
        lltype::LowLevelValue::Ptr(rtti) => Ok(*rtti),
        other => Err(TyperError::message(format!(
            "ll_runtime_type_info: expected Ptr rtti, got {other:?}"
        ))),
    }
}

/// RPython `ll_inst_type(obj)`.
pub fn ll_inst_type(obj: Option<&_ptr>) -> Result<Option<_ptr>, TyperError> {
    match obj {
        Some(ptr) if ptr.nonzero() => ll_type(ptr).map(Some),
        _ => Ok(None),
    }
}

fn signed_vtable_field(ptr: &_ptr, field: &str, ctx: &str) -> Result<i64, TyperError> {
    match ptr.getattr(field).map_err(TyperError::message)? {
        lltype::LowLevelValue::Signed(value) => Ok(value),
        other => Err(TyperError::message(format!(
            "{ctx}: {field} must be Signed, got {other:?}"
        ))),
    }
}

/// RPython `ll_issubclass(subcls, cls)` (`rclass.py`).
pub fn ll_issubclass(subcls: &_ptr, cls: &_ptr) -> Result<bool, TyperError> {
    let cls_min = signed_vtable_field(cls, "subclassrange_min", "ll_issubclass")?;
    let subcls_min = signed_vtable_field(subcls, "subclassrange_min", "ll_issubclass")?;
    let cls_max = signed_vtable_field(cls, "subclassrange_max", "ll_issubclass")?;
    Ok(cls_min <= subcls_min && subcls_min < cls_max)
}

/// RPython `ll_issubclass_const(subcls, minid, maxid)`.
pub fn ll_issubclass_const(subcls: &_ptr, minid: i64, maxid: i64) -> Result<bool, TyperError> {
    let subcls_min = signed_vtable_field(subcls, "subclassrange_min", "ll_issubclass_const")?;
    Ok(minid <= subcls_min && subcls_min < maxid)
}

/// RPython `ll_isinstance(obj, cls)` (`rclass.py`).
pub fn ll_isinstance(obj: &_ptr, cls: &_ptr) -> Result<bool, TyperError> {
    if !obj.nonzero() {
        return Ok(false);
    }
    let obj_cls = match obj.getattr("typeptr").map_err(TyperError::message)? {
        lltype::LowLevelValue::Ptr(typeptr) => *typeptr,
        other => {
            return Err(TyperError::message(format!(
                "ll_isinstance: typeptr must be Ptr, got {other:?}"
            )));
        }
    };
    ll_issubclass(&obj_cls, cls)
}

/// RPython `ll_both_none(ins1, ins2)` (`rclass.py`).
pub fn ll_both_none(ins1: &_ptr, ins2: &_ptr) -> bool {
    !ins1.nonzero() && !ins2.nonzero()
}

/// RPython `make_ll_isinstance(rtyper, cls)` (`rclass.py`).
pub fn make_ll_isinstance(
    rtyper: &RPythonTyper,
    cls_ptr: &_ptr,
    obj_lltype: &LowLevelType,
) -> Result<(LowLevelFunction, LowLevelFunction), TyperError> {
    crate::translator::rtyper::rtyper::make_ll_isinstance(rtyper, cls_ptr, obj_lltype)
}

/// RPython `feedllattr(inst, name, llvalue)`.
pub fn feedllattr(
    inst: &mut _ptr,
    name: &str,
    llvalue: lltype::LowLevelValue,
) -> Result<(), TyperError> {
    let field = format!("inst_{name}");
    let mut p = inst.clone();
    loop {
        if p.setattr(&field, llvalue.clone()).is_ok() {
            return Ok(());
        }
        match p.getattr("super") {
            Ok(lltype::LowLevelValue::Ptr(super_ptr)) => p = *super_ptr,
            _ => break,
        }
    }
    Err(TyperError::message(format!(
        "feedllattr: {:?} has no field {name}",
        inst._TYPE
    )))
}

/// RPython `declare_type_for_typeptr(vtable, TYPE)`.
pub fn declare_type_for_typeptr(_vtable: &_ptr, _TYPE: &LowLevelType) -> Result<(), TyperError> {
    Err(rclass_deferred("declare_type_for_typeptr"))
}

/// RPython `class Entry(ExtRegistryEntry)` for `declare_type_for_typeptr`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry;

/// Adapter from a converted [`Constant`] (produced by
/// [`crate::translator::rtyper::rmodel::Repr::convert_const`] /
/// [`crate::translator::rtyper::rmodel::Repr::convert_desc_or_const`])
/// to the
/// [`crate::translator::rtyper::lltypesystem::lltype::LowLevelValue`]
/// shape that [`_ptr::setattr`](crate::translator::rtyper::lltypesystem::lltype::_ptr::setattr)
/// consumes for vtable slot writes (rclass.py `setattr(vtable,
/// mangled_name, llvalue)`).
///
/// Upstream Python performs no explicit conversion — the `Constant`
/// object itself is stored through the live lltype container's
/// `__setattr__`. The Rust port has separate `LowLevelValue` and
/// `ConstValue` enums, so this helper picks the variant that matches
/// the target field type carried on `c.concretetype`.
///
/// Scope: covers the concrete shapes needed by `setup_vtable`'s
/// vtable-field writes — `Void`, `Signed`, `Bool`, `Float`, and `Ptr`.
/// `Unsigned` / `Char` / `UniChar` / `SingleFloat` / `LongFloat` etc.
/// surface as `TyperError` until their repr ports land.
pub(crate) fn constant_to_lowlevel_value(
    c: &crate::flowspace::model::Constant,
) -> Result<lltype::LowLevelValue, TyperError> {
    let target = c.concretetype.as_ref().ok_or_else(|| {
        TyperError::message(format!(
            "constant_to_lowlevel_value: constant {:?} lacks concretetype",
            c.value
        ))
    })?;
    match (target, &c.value) {
        (LowLevelType::Void, _) => Ok(lltype::LowLevelValue::Void),
        (LowLevelType::Signed, ConstValue::Int(n)) => Ok(lltype::LowLevelValue::Signed(*n)),
        (LowLevelType::Signed, ConstValue::Bool(b)) => {
            Ok(lltype::LowLevelValue::Signed(i64::from(*b)))
        }
        (LowLevelType::Bool, ConstValue::Bool(b)) => Ok(lltype::LowLevelValue::Bool(*b)),
        (LowLevelType::Float, ConstValue::Float(bits)) => Ok(lltype::LowLevelValue::Float(*bits)),
        (LowLevelType::Address, ConstValue::LLAddress(addr)) => {
            Ok(lltype::LowLevelValue::Address(addr.clone()))
        }
        (LowLevelType::Ptr(_), ConstValue::LLPtr(ptr)) => {
            Ok(lltype::LowLevelValue::Ptr(ptr.clone()))
        }
        _ => Err(TyperError::message(format!(
            "constant_to_lowlevel_value: unsupported ConstValue→LowLevelValue for target {target:?} and value {:?}",
            c.value
        ))),
    }
}

// rclass.py — ClassRepr (classdef != None flavour).

/// RPython `class ClassRepr(Repr)` (rclass.py).
///
/// Readonly class attrs populate `clsfields` / `allmethods`, and
/// `extra_access_sets` populate `pbcfields`, matching the attr walk in
/// rclass.py. Method-valued attrs flow through
/// `prepare_method()` before repr lookup so method-descriptor PBCs are
/// rewritten to function PBCs like upstream.
#[derive(Debug)]
pub struct ClassRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:193). Weak because
    /// `ClassDef.repr` holds the strong repr cache entry, and the
    /// back-edge mirrors R1's `Weak<RPythonAnnotator>` on
    /// `RPythonTyper.annotator`.
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef = classdef` (rclass.py:194). Always
    /// non-None for `ClassRepr`; `classdef is None` routes through
    /// [`RootClassRepr`] per upstream's `class
    /// RootClassRepr(ClassRepr)` override at rclass.py.
    classdef: Rc<RefCell<ClassDef>>,
    /// RPython `self.vtable_type = lltype.ForwardReference()`
    /// (rclass.py:195). Stored as `LowLevelType::ForwardReference`; the
    /// inner `Arc<Mutex<Option<LowLevelType>>>` target is shared with
    /// [`ClassRepr::lowleveltype`]'s Ptr target so resolving one also
    /// resolves the other.
    vtable_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.vtable_type)`
    /// (rclass.py:196).
    lowleveltype: LowLevelType,
    /// RPython `self.clsfields = clsfields` (rclass.py). `{name:
    /// (mangled_name, r)}` dict holding the class-level attribute
    /// reprs, populated by `_setup_repr`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    clsfields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.pbcfields = pbcfields` (rclass.py).
    /// Upstream keys by `(access_set, attr)`; Rust stores the
    /// access-set identity pointer alongside `attr`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pbcfields: RefCell<HashMap<(usize, String), (String, Arc<dyn Repr>)>>,
    /// RPython `self.allmethods = allmethods` (rclass.py).
    /// `{name: True}` set-as-dict upstream; Rust keeps the dict shape
    /// to remain line-by-line portable even though `HashSet<String>`
    /// would be idiomatic.
    allmethods: RefCell<HashMap<String, bool>>,
    /// RPython `self.rbase = getclassrepr(...)` (rclass.py).
    rbase: RefCell<Option<ClassReprArc>>,
    /// RPython `self.vtable = None` (rclass.py:284). Lazily filled by
    /// [`Self::init_vtable`] and read by [`Self::getvtable`]. Stores the
    /// solid vtable pointer (`malloc(vtable_type, immortal=True)`).
    vtable: RefCell<Option<_ptr>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl ClassRepr {
    /// RPython `ClassRepr.__init__(self, rtyper, classdef)`
    /// (rclass.py).
    ///
    /// Allocates a fresh `ForwardReference` for the vtable container and
    /// stores `Ptr(self.vtable_type)` as the Repr's low-level type. The
    /// `ForwardReference`'s `target: Arc<Mutex<_>>` is cloned into the
    /// `Ptr.TO` variant so both fields observe the same resolution once
    /// `_setup_repr` lands the Struct body.
    pub fn new(rtyper: &Rc<RPythonTyper>, classdef: &Rc<RefCell<ClassDef>>) -> Self {
        let fwd = ForwardReference::new();
        let fwd_for_ptr = fwd.clone();
        let vtable_type = LowLevelType::ForwardReference(Box::new(fwd));
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd_for_ptr),
        }));
        ClassRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: classdef.clone(),
            vtable_type,
            lowleveltype,
            clsfields: RefCell::new(HashMap::new()),
            pbcfields: RefCell::new(HashMap::new()),
            allmethods: RefCell::new(HashMap::new()),
            rbase: RefCell::new(None),
            vtable: RefCell::new(None),
            state: ReprState::new(),
        }
    }

    /// RPython `ClassRepr.classdef` (rclass.py).
    pub fn classdef(&self) -> Rc<RefCell<ClassDef>> {
        self.classdef.clone()
    }

    /// True once [`Self::init_vtable`] has built the solid vtable. A
    /// materialized vtable means this class's `subclassrange_min/max` has
    /// been (or may have been) baked into an immutable jitcode constant, so
    /// `normalizecalls::assign_inheritance_ids` must not re-resolve it to a
    /// different range.
    pub fn is_vtable_materialized(&self) -> bool {
        self.vtable.borrow().is_some()
    }

    /// RPython `ClassRepr.vtable_type` (rclass.py). Exposed so
    /// `ClassRepr._setup_repr` (on a child class) can borrow the parent
    /// vtable_type when building its `('super', rbase.vtable_type)`
    /// Struct entry.
    pub fn vtable_type(&self) -> &LowLevelType {
        &self.vtable_type
    }

    /// Read-only view of the class-level `clsfields` dict populated by
    /// `_setup_repr`. RPython `self.clsfields` (rclass.py).
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub fn clsfields(&self) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.clsfields.borrow()
    }

    /// Read-only view of the class-level `allmethods` dict. RPython
    /// `self.allmethods` (rclass.py:283).
    pub fn allmethods(&self) -> std::cell::Ref<'_, HashMap<String, bool>> {
        self.allmethods.borrow()
    }

    fn prepare_method(&self, s_value: &SomeValue) -> Result<Option<SomeValue>, TyperError> {
        let SomeValue::PBC(s_pbc) = s_value else {
            return Ok(None);
        };
        let kind = s_pbc
            .get_kind()
            .map_err(|err| TyperError::message(err.to_string()))?;
        if kind != DescKind::Method {
            return Ok(None);
        }
        let s_unbound = ClassDef::lookup_filter(&self.classdef, s_pbc, None, &BTreeMap::new())
            .map_err(|err| TyperError::message(err.to_string()))?;
        let SomeValue::PBC(s_unbound) = s_unbound else {
            return Err(TyperError::message(
                "ClassRepr.prepare_method expected lookup_filter() to return SomePBC",
            ));
        };
        let funcdescs: Vec<DescEntry> = s_unbound
            .descriptions
            .values()
            .map(|entry| {
                let method = entry.as_method().ok_or_else(|| {
                    TyperError::message(
                        "ClassRepr.prepare_method expected MethodDesc entries after lookup_filter",
                    )
                })?;
                Ok(DescEntry::Func(method.borrow().funcdesc.clone()))
            })
            .collect::<Result<_, TyperError>>()?;
        // rclass.py — `return annmodel.SomePBC(funcdescs)`. The
        // upstream constructor uses the `can_be_None=False` default; do
        // not propagate `s_unbound.can_be_none` (would alter the
        // nullable-method-PBC repr/key shape).
        Ok(Some(SomeValue::PBC(SomePBC::new(funcdescs, false))))
    }

    /// RPython `ClassRepr.setup_vtable(self, vtable, r_parentcls)`
    /// (rclass.py).
    ///
    /// ```python
    /// def setup_vtable(self, vtable, r_parentcls):
    ///     def assign(mangled_name, value):
    ///         if value is None:
    ///             llvalue = r.special_uninitialized_value()
    ///             if llvalue is None:
    ///                 return
    ///         else:
    ///             if (isinstance(value, Constant) and
    ///                     isinstance(value.value, staticmethod)):
    ///                 value = Constant(value.value.__get__(42))
    ///             llvalue = r.convert_desc_or_const(value)
    ///         setattr(vtable, mangled_name, llvalue)
    ///
    ///     for fldname in r_parentcls.clsfields:
    ///         mangled_name, r = r_parentcls.clsfields[fldname]
    ///         if r.lowleveltype is Void:
    ///             continue
    ///         value = self.classdef.classdesc.read_attribute(fldname, None)
    ///         assign(mangled_name, value)
    ///     for (access_set, attr), (mangled_name, r) in r_parentcls.pbcfields.items():
    ///         if self.classdef.classdesc not in access_set.descs:
    ///             continue
    ///         if r.lowleveltype is Void:
    ///             continue
    ///         attrvalue = self.classdef.classdesc.read_attribute(attr, None)
    ///         assign(mangled_name, attrvalue)
    /// ```
    ///
    /// `vtable` points at the vtable sub-struct for the level
    /// `r_parentcls` describes (rclass.py walks the super chain
    /// via [`Self::init_vtable`]). Upstream calls `setattr` on the
    /// live `_struct` container; the Rust port routes through
    /// [`_ptr::setattr`] after mapping the converted
    /// [`crate::flowspace::model::Constant`] to a
    /// [`lltype::LowLevelValue`] via [`constant_to_lowlevel_value`].
    pub fn setup_vtable(
        &self,
        vtable: &mut _ptr,
        r_parentcls: &Arc<ClassRepr>,
    ) -> Result<(), TyperError> {
        // Backward-compatible shim — leaf-level write at empty path.
        // [`Self::init_vtable`] uses [`Self::setup_vtable_at_path`]
        // directly to address nested levels.
        self.setup_vtable_at_path(vtable, &[], r_parentcls)
    }

    /// Path-aware variant of [`Self::setup_vtable`] used by
    /// [`Self::init_vtable`] to mutate nested vtable substructs.
    /// `path` is a sequence of `"super"` segments naming the level to
    /// write — empty for the leaf vtable, `["super"]` for the parent
    /// level, and so on. Each `assign` call routes through
    /// [`_ptr::setattr_at_path`] so writes alias the original allocation
    /// instead of being lost on a detached substruct copy returned by
    /// `_ptr.getattr("super")`.
    pub fn setup_vtable_at_path(
        &self,
        vtable: &mut _ptr,
        path: &[&str],
        r_parentcls: &Arc<ClassRepr>,
    ) -> Result<(), TyperError> {
        let classdesc = self.classdef.borrow().classdesc.clone();
        let classdesc_key = classdesc.borrow().identity;

        // upstream: `for fldname in r_parentcls.clsfields`.
        #[expect(
            clippy::type_complexity,
            reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
        )]
        let clsfields_snapshot: Vec<(String, (String, Arc<dyn Repr>))> = r_parentcls
            .clsfields
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for (fldname, (mangled_name, r)) in &clsfields_snapshot {
            if matches!(r.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            let value = crate::annotator::classdesc::ClassDesc::read_attribute(&classdesc, fldname);
            self.setup_vtable_assign(vtable, path, mangled_name, r, value)?;
        }

        // upstream: extra PBC attributes —
        // `for (access_set, attr), (mangled_name, r) in r_parentcls.pbcfields.items()`.
        #[expect(
            clippy::type_complexity,
            reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
        )]
        let pbcfields_snapshot: Vec<((usize, String), (String, Arc<dyn Repr>))> = r_parentcls
            .pbcfields
            .borrow()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        for ((access_set_id, attr), (mangled_name, r)) in &pbcfields_snapshot {
            // upstream: `if self.classdef.classdesc not in access_set.descs: continue`.
            let Some(access_set) = r_parentcls
                .classdef
                .borrow()
                .extra_access_sets
                .values()
                .find_map(|(family, _, _)| {
                    (class_attr_family_key(family) == *access_set_id).then(|| family.clone())
                })
            else {
                continue;
            };
            if !access_set.borrow().descs.contains_key(&classdesc_key) {
                continue;
            }
            if matches!(r.lowleveltype(), LowLevelType::Void) {
                continue;
            }
            let attrvalue =
                crate::annotator::classdesc::ClassDesc::read_attribute(&classdesc, attr);
            self.setup_vtable_assign(vtable, path, mangled_name, r, attrvalue)?;
        }
        Ok(())
    }

    /// Helper that mirrors the nested `def assign(...)` closure in
    /// upstream `ClassRepr.setup_vtable` (rclass.py). Routes
    /// through [`_ptr::setattr_at_path`] so substruct writes propagate
    /// back to the parent allocation.
    fn setup_vtable_assign(
        &self,
        vtable: &mut _ptr,
        path: &[&str],
        mangled_name: &str,
        r: &Arc<dyn Repr>,
        value: Option<crate::annotator::classdesc::ClassDictEntry>,
    ) -> Result<(), TyperError> {
        let llvalue = match value {
            None => {
                // upstream: `llvalue = r.special_uninitialized_value()`
                // + `if llvalue is None: return`.
                let Some(raw) = r.special_uninitialized_value() else {
                    return Ok(());
                };
                let constant = crate::flowspace::model::Constant::with_concretetype(
                    raw,
                    r.lowleveltype().clone(),
                );
                constant_to_lowlevel_value(&constant)?
            }
            Some(entry) => {
                // upstream: staticmethod unwrap — `if isinstance(value,
                // Constant) and isinstance(value.value, staticmethod):
                // value = Constant(value.value.__get__(42))`.
                let entry = match entry {
                    crate::annotator::classdesc::ClassDictEntry::Constant(c) => {
                        let c = match &c.value {
                            ConstValue::HostObject(host) if host.is_staticmethod() => {
                                let Some(func) = host.staticmethod_func().cloned() else {
                                    return Err(TyperError::message(
                                        "setup_vtable_assign: staticmethod without underlying func",
                                    ));
                                };
                                crate::flowspace::model::Constant::new(ConstValue::HostObject(func))
                            }
                            _ => c,
                        };
                        DescOrConst::Const(c)
                    }
                    crate::annotator::classdesc::ClassDictEntry::Desc(d) => DescOrConst::Desc(d),
                };
                let converted = r.convert_desc_or_const(&entry)?;
                constant_to_lowlevel_value(&converted)?
            }
        };
        if path.is_empty() {
            vtable
                .setattr(mangled_name, llvalue)
                .map_err(TyperError::message)?;
        } else {
            vtable
                .setattr_at_path(path, mangled_name, llvalue)
                .map_err(TyperError::message)?;
        }
        Ok(())
    }

    /// RPython `ClassRepr.init_vtable(self)` (rclass.py).
    ///
    /// ```python
    /// def init_vtable(self):
    ///     """Create the actual vtable"""
    ///     self.vtable = malloc(self.vtable_type, immortal=True)
    ///     vtable_part = self.vtable
    ///     r_parentcls = self
    ///     while r_parentcls.classdef is not None:
    ///         self.setup_vtable(vtable_part, r_parentcls)
    ///         vtable_part = vtable_part.super
    ///         r_parentcls = r_parentcls.rbase
    ///     self.fill_vtable_root(vtable_part)
    /// ```
    ///
    /// Walks the super-chain from `self` toward the root, invoking
    /// `setup_vtable` for each ClassRepr level and finishing with
    /// `fill_vtable_root` on the root vtable slice (the OBJECT_VTABLE
    /// portion). Idempotent via the `vtable: Option<_ptr>` cache —
    /// callers go through [`Self::getvtable`] instead of calling this
    /// directly.
    pub fn init_vtable(self: &Arc<Self>) -> Result<(), TyperError> {
        // upstream: `self.vtable = malloc(self.vtable_type, immortal=True)`.
        let vtable_lltype = match self.vtable_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "ClassRepr.init_vtable: vtable_type ForwardReference not resolved \
                     (call setup() first)",
                )
            })?,
            other => other,
        };
        let mut vtable_root = crate::translator::rtyper::lltypesystem::lltype::malloc(
            vtable_lltype,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Gc,
            true,
        )
        .map_err(TyperError::message)?;

        // Walk the super-chain via increasing `path`, calling
        // `setup_vtable_at_path(vtable_root, path, r_parentcls)` at
        // each level. Upstream's `vtable_part = vtable_part.super`
        // re-uses the same allocation by reference; the Rust port
        // models that with a path-tracked address into the root
        // allocation (see [`_ptr::setattr_at_path`]).
        let mut path: Vec<&str> = Vec::new();
        let mut r_parentcls: ClassReprArc = ClassReprArc::Inst(self.clone());
        while let ClassReprArc::Inst(parent_inst) = &r_parentcls {
            // Receiver is `self` (the leaf class) — values come from
            // `self.classdef.classdesc` while the field shape comes from
            // `r_parentcls.clsfields`. Upstream rclass.py:302 reads
            // `self.setup_vtable(vtable_part, r_parentcls)`.
            self.setup_vtable_at_path(&mut vtable_root, &path, parent_inst)?;
            // upstream: `vtable_part = vtable_part.super`.
            path.push("super");
            // upstream: `r_parentcls = r_parentcls.rbase`.
            let next = parent_inst.rbase.borrow().clone().ok_or_else(|| {
                TyperError::message("ClassRepr.init_vtable: rbase missing — call setup() first")
            })?;
            r_parentcls = next;
        }
        // upstream: `self.fill_vtable_root(vtable_part)` — at the
        // OBJECT_VTABLE level, addressed by the accumulated `path`.
        self.fill_vtable_root_at_path(&mut vtable_root, &path)?;

        // Cache and return.
        *self.vtable.borrow_mut() = Some(vtable_root);
        Ok(())
    }

    /// RPython `ClassRepr.fill_vtable_root(self, vtable)` (rclass.py).
    ///
    /// ```python
    /// def fill_vtable_root(self, vtable):
    ///     if self.classdef is not None:
    ///         vtable.subclassrange_min = self.classdef.minid
    ///         vtable.subclassrange_max = self.classdef.maxid
    ///     else:
    ///         vtable.subclassrange_min = 0
    ///         vtable.subclassrange_max = sys.maxint
    ///     rinstance = getinstancerepr(self.rtyper, self.classdef)
    ///     rinstance.setup()
    ///     if rinstance.gcflavor == 'gc':
    ///         vtable.rtti = getRuntimeTypeInfo(rinstance.object_type)
    ///     ...
    ///     vtable.name = alloc_array_name(name)
    ///     if hasattr(self.classdef, 'my_instantiate_graph'):
    ///         vtable.instantiate = self.rtyper.getcallable(graph)
    /// ```
    ///
    /// `my_instantiate_graph` is a `FunctionGraph`. `getcallable`'s
    /// non-sandbox body is `getfunctionptr` over `bindingrepr`; the
    /// graph's `func` is unset, so the sandbox external-name branch
    /// does not apply.
    pub fn fill_vtable_root(&self, vtable: &mut _ptr) -> Result<(), TyperError> {
        // Backward-compatible shim — leaf-level write at empty path.
        self.fill_vtable_root_at_path(vtable, &[])
    }

    /// Path-aware variant of [`Self::fill_vtable_root`] used by
    /// [`Self::init_vtable`] to address the OBJECT_VTABLE substruct
    /// nested inside the per-class vtable allocation. `path` is the
    /// chain of `"super"` segments leading from the leaf vtable to
    /// the OBJECT_VTABLE level.
    pub fn fill_vtable_root_at_path(
        &self,
        vtable: &mut _ptr,
        path: &[&str],
    ) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.fill_vtable_root: RPythonTyper weak ref expired")
        })?;
        let classdef = self.classdef.clone();
        // upstream: subclassrange_min / subclassrange_max.
        let (minid, maxid) = {
            let cd = classdef.borrow();
            let minid = cd.minid.ok_or_else(|| {
                TyperError::message(
                    "ClassRepr.fill_vtable_root: classdef.minid not set — \
                     run normalizecalls.assign_inheritance_ids first",
                )
            })?;
            let maxid = cd.maxid.ok_or_else(|| {
                TyperError::message(
                    "ClassRepr.fill_vtable_root: classdef.maxid not set — \
                     run normalizecalls.assign_inheritance_ids first",
                )
            })?;
            (minid, maxid)
        };
        let setattr_path =
            |vtable: &mut _ptr, name: &str, val: lltype::LowLevelValue| -> Result<(), TyperError> {
                if path.is_empty() {
                    vtable.setattr(name, val).map_err(TyperError::message)
                } else {
                    vtable
                        .setattr_at_path(path, name, val)
                        .map_err(TyperError::message)
                }
            };
        setattr_path(
            vtable,
            "subclassrange_min",
            lltype::LowLevelValue::Signed(minid),
        )?;
        setattr_path(
            vtable,
            "subclassrange_max",
            lltype::LowLevelValue::Signed(maxid),
        )?;

        // upstream: `rinstance = getinstancerepr(self.rtyper, self.classdef);
        //           rinstance.setup();
        //           if rinstance.gcflavor == 'gc':
        //               vtable.rtti = getRuntimeTypeInfo(rinstance.object_type)`.
        let rinstance = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)?;
        Repr::setup(rinstance.as_ref() as &dyn Repr)?;
        if rinstance.gcflavor() == Flavor::Gc {
            // `object_type` for non-None classdefs is a
            // ForwardReference that `_setup_repr.r#become` resolves to
            // a `Struct`. `getRuntimeTypeInfo` only accepts `Struct`
            // directly, so resolve before passing.
            let object_lltype = match rinstance.object_type().clone() {
                LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                    TyperError::message(
                        "ClassRepr.fill_vtable_root: InstanceRepr.object_type \
                         ForwardReference not resolved",
                    )
                })?,
                other => other,
            };
            let rtti =
                crate::translator::rtyper::lltypesystem::lltype::getRuntimeTypeInfo(&object_lltype)
                    .map_err(TyperError::message)?;
            setattr_path(vtable, "rtti", lltype::LowLevelValue::Ptr(Box::new(rtti)))?;
        }
        // upstream rclass.py:351-355 — root uses "object"; non-root
        // classdefs use `shortname`, not the fully-qualified name.
        let class_shortname = classdef.borrow().shortname.clone();
        let name_ptr =
            crate::translator::rtyper::lltypesystem::rstr::alloc_array_name(&class_shortname)
                .map_err(TyperError::message)?;
        setattr_path(
            vtable,
            "name",
            lltype::LowLevelValue::Ptr(Box::new(name_ptr)),
        )?;
        // upstream: `if hasattr(self.classdef, 'my_instantiate_graph')`.
        if let Some(graph) = classdef.borrow().my_instantiate_graph.clone() {
            let funcptr =
                crate::translator::rtyper::lltypesystem::lltype::getfunctionptr(&graph, |v| {
                    rtyper
                        .bindingrepr(v)
                        .map(|repr| repr.lowleveltype().clone())
                })?;
            setattr_path(
                vtable,
                "instantiate",
                lltype::LowLevelValue::Ptr(Box::new(funcptr)),
            )?;
        }
        Ok(())
    }

    /// RPython `ClassRepr.getvtable(self)` (rclass.py).
    ///
    /// ```python
    /// def getvtable(self):
    ///     if self.vtable is None:
    ///         self.init_vtable()
    ///     return cast_vtable_to_typeptr(self.vtable)
    /// ```
    pub fn getvtable(self: &Arc<Self>) -> Result<_ptr, TyperError> {
        if self.vtable.borrow().is_none() {
            self.init_vtable()?;
        }
        let vtable = self
            .vtable
            .borrow()
            .clone()
            .expect("init_vtable post-condition: vtable is Some");
        Ok(cast_vtable_to_typeptr(vtable))
    }

    /// RPython `ClassRepr.getruntime(self, expected_type)` (rclass.py).
    ///
    /// ```python
    /// def getruntime(self, expected_type):
    ///     assert expected_type == CLASSTYPE
    ///     return self.getvtable()
    /// ```
    pub fn getruntime(self: &Arc<Self>, expected_type: &LowLevelType) -> Result<_ptr, TyperError> {
        if expected_type != &CLASSTYPE.clone() {
            return Err(TyperError::message(format!(
                "ClassRepr.getruntime: expected CLASSTYPE, got {expected_type:?}"
            )));
        }
        self.getvtable()
    }

    /// RPython `ClassRepr.fromtypeptr(self, vcls, llops)`
    /// (rclass.py).
    ///
    /// ```python
    /// def fromtypeptr(self, vcls, llops):
    ///     self.setup()
    ///     castable(self.lowleveltype, vcls.concretetype)  # sanity check
    ///     return llops.genop('cast_pointer', [vcls],
    ///                        resulttype=self.lowleveltype)
    /// ```
    ///
    /// The upstream `castable` sanity check is deferred — `genop` will
    /// surface a TypeError on a malformed cast at lower codegen, and
    /// our static `castable` helper has not landed.
    pub fn fromtypeptr(
        self: &Arc<Self>,
        vcls: Hlvalue,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        Repr::setup(self.as_ref() as &dyn Repr)?;
        Ok(llops
            .genop(
                "cast_pointer",
                vec![vcls],
                GenopResult::LLType(self.lowleveltype.clone()),
            )
            .expect("cast_pointer with non-Void resulttype yields a Variable"))
    }

    /// RPython `ClassRepr.getclsfield(self, vcls, attr, llops)`
    /// (rclass.py).
    ///
    /// ```python
    /// def getclsfield(self, vcls, attr, llops):
    ///     if attr in self.clsfields:
    ///         mangled_name, r = self.clsfields[attr]
    ///         v_vtable = self.fromtypeptr(vcls, llops)
    ///         cname = inputconst(Void, mangled_name)
    ///         return llops.genop('getfield', [v_vtable, cname], resulttype=r)
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         return self.rbase.getclsfield(vcls, attr, llops)
    /// ```
    pub fn getclsfield(
        self: &Arc<Self>,
        vcls: Hlvalue,
        attr: &str,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        if let Some((mangled_name, r)) = self.clsfields.borrow().get(attr).cloned() {
            let v_vtable = self.fromtypeptr(vcls, llops)?;
            let cname =
                Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
            return Ok(llops
                .genop(
                    "getfield",
                    vec![Hlvalue::Variable(v_vtable), Hlvalue::Constant(cname)],
                    GenopResult::LLType(r.lowleveltype().clone()),
                )
                .expect("getfield with non-Void result yields a Variable"));
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        // ClassRepr always has classdef != None, so route to rbase.
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message(format!(
                "ClassRepr.getclsfield({attr:?}) on class {:?}: \
                 rbase missing — call setup() first",
                self.classdef.borrow().name
            ))
        })?;
        match rbase {
            ClassReprArc::Inst(inst) => inst.getclsfield(vcls, attr, llops),
            ClassReprArc::Root(root) => root.getclsfield(vcls, attr, llops),
        }
    }

    /// RPython `ClassRepr.setclsfield(self, vcls, attr, vvalue, llops)`
    /// (rclass.py).
    ///
    /// ```python
    /// def setclsfield(self, vcls, attr, vvalue, llops):
    ///     if attr in self.clsfields:
    ///         mangled_name, r = self.clsfields[attr]
    ///         v_vtable = self.fromtypeptr(vcls, llops)
    ///         cname = inputconst(Void, mangled_name)
    ///         llops.genop('setfield', [v_vtable, cname, vvalue])
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         self.rbase.setclsfield(vcls, attr, vvalue, llops)
    /// ```
    pub fn setclsfield(
        self: &Arc<Self>,
        vcls: Hlvalue,
        attr: &str,
        vvalue: Hlvalue,
        llops: &mut LowLevelOpList,
    ) -> Result<(), TyperError> {
        if let Some((mangled_name, _r)) = self.clsfields.borrow().get(attr).cloned() {
            let v_vtable = self.fromtypeptr(vcls, llops)?;
            let cname =
                Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
            llops.genop(
                "setfield",
                vec![
                    Hlvalue::Variable(v_vtable),
                    Hlvalue::Constant(cname),
                    vvalue,
                ],
                GenopResult::Void,
            );
            return Ok(());
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("ClassRepr.setclsfield: rbase missing — call setup() first")
        })?;
        match rbase {
            ClassReprArc::Inst(inst) => inst.setclsfield(vcls, attr, vvalue, llops),
            ClassReprArc::Root(root) => root.setclsfield(vcls, attr, vvalue, llops),
        }
    }

    /// RPython `ClassRepr.getpbcfield(self, vcls, access_set, attr,
    /// llops)` (rclass.py).
    ///
    /// ```python
    /// def getpbcfield(self, vcls, access_set, attr, llops):
    ///     if (access_set, attr) not in self.pbcfields:
    ///         raise TyperError("internal error: missing PBC field")
    ///     mangled_name, r = self.pbcfields[access_set, attr]
    ///     v_vtable = self.fromtypeptr(vcls, llops)
    ///     cname = inputconst(Void, mangled_name)
    ///     return llops.genop('getfield', [v_vtable, cname], resulttype=r)
    /// ```
    ///
    /// The Rust port keys `pbcfields` on `(access_set_id: usize, attr:
    /// String)` so callers pass the access-set pointer-identity hash
    /// (via [`class_attr_family_key`]).
    pub fn getpbcfield(
        self: &Arc<Self>,
        vcls: Hlvalue,
        access_set_id: usize,
        attr: &str,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        let (mangled_name, r) = self
            .pbcfields
            .borrow()
            .get(&(access_set_id, attr.to_string()))
            .cloned()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "ClassRepr.getpbcfield: internal error: missing PBC field \
                     (access_set={access_set_id}, attr={attr:?})"
                ))
            })?;
        let v_vtable = self.fromtypeptr(vcls, llops)?;
        let cname =
            Constant::with_concretetype(ConstValue::byte_str(mangled_name), LowLevelType::Void);
        Ok(llops
            .genop(
                "getfield",
                vec![Hlvalue::Variable(v_vtable), Hlvalue::Constant(cname)],
                GenopResult::LLType(r.lowleveltype().clone()),
            )
            .expect("getfield with non-Void result yields a Variable"))
    }

    /// RPython `ClassRepr.rtype_issubtype(self, hop)` (rclass.py).
    ///
    /// ```python
    /// def rtype_issubtype(self, hop):
    ///     class_repr = get_type_repr(self.rtyper)
    ///     v_cls1, v_cls2 = hop.inputargs(class_repr, class_repr)
    ///     if isinstance(v_cls2, Constant):
    ///         cls2 = v_cls2.value
    ///         minid = hop.inputconst(Signed, cls2.subclassrange_min)
    ///         maxid = hop.inputconst(Signed, cls2.subclassrange_max)
    ///         return hop.gendirectcall(ll_issubclass_const, v_cls1, minid, maxid)
    ///     else:
    ///         v_cls1, v_cls2 = hop.inputargs(class_repr, class_repr)
    ///         return hop.gendirectcall(ll_issubclass, v_cls1, v_cls2)
    /// ```
    ///
    /// Both branches dispatch to a low-level helper graph minted via
    /// [`RPythonTyper::lowlevel_helper_function`]: `ll_issubclass`
    /// (rclass.py) for the variable case and
    /// `ll_issubclass_const` (rclass.py) for the constant
    /// case. The helper bodies emit a single `int_between` op against
    /// `cls.subclassrange_min` / `cls.subclassrange_max`.
    /// Inherent method body — receiver is `&self` so the
    /// [`Repr::rtype_issubtype`] trait override can forward to it
    /// without needing the `Arc<Self>` smart pointer (no upstream
    /// caller depends on the Arc handle here; only `self.rtyper.upgrade()`
    /// is touched).
    pub fn rtype_issubtype(
        &self,
        hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    ) -> RTypeResult {
        rtype_issubtype_helper(&self.rtyper, hop, "ClassRepr.rtype_issubtype")
    }
}

/// Shared body of `AbstractClassRepr.rtype_issubtype` (rclass.py).
///
/// Upstream both `ClassRepr` and `RootClassRepr` inherit this method
/// from `AbstractClassRepr`. The Rust port has them as separate
/// structs, so the body is hoisted into a free helper that both
/// `Repr::rtype_issubtype` overrides forward to. The only
/// per-receiver state the body uses is the `rtyper` weak ref, so the
/// helper takes that explicitly.
fn rtype_issubtype_helper(
    rtyper_weak: &Weak<RPythonTyper>,
    hop: &crate::translator::rtyper::rtyper::HighLevelOp,
    caller: &'static str,
) -> RTypeResult {
    use crate::translator::rtyper::rtyper::ConvertedTo;
    let rtyper = rtyper_weak
        .upgrade()
        .ok_or_else(|| TyperError::message(format!("{caller}: rtyper weak ref expired")))?;
    let class_repr = get_type_repr(&rtyper)?;
    let class_repr_dyn: &dyn Repr = class_repr.as_ref();
    let v_args = hop.inputargs(vec![
        ConvertedTo::Repr(class_repr_dyn),
        ConvertedTo::Repr(class_repr_dyn),
    ])?;
    let v_cls1 = v_args[0].clone();
    let v_cls2 = v_args[1].clone();

    // upstream `if isinstance(v_cls2, Constant)`. The constant
    // branch unpacks `cls2.value.subclassrange_{min,max}` so we
    // need an `_ptr` carrying `subclassrange_min/max` Signed
    // fields (the OBJECT_VTABLE struct, after fill_vtable_root).
    if let Hlvalue::Constant(c) = &v_cls2 {
        let ConstValue::LLPtr(c_ptr) = &c.value else {
            return Err(TyperError::message(format!(
                "{caller}: constant cls2 must carry an _ptr, got {:?}",
                c.value
            )));
        };
        let min_val = c_ptr
            .getattr("subclassrange_min")
            .map_err(TyperError::message)?;
        let max_val = c_ptr
            .getattr("subclassrange_max")
            .map_err(TyperError::message)?;
        let lltype::LowLevelValue::Signed(min_n) = min_val else {
            return Err(TyperError::message(format!(
                "{caller}: subclassrange_min not Signed, got {min_val:?}"
            )));
        };
        let lltype::LowLevelValue::Signed(max_n) = max_val else {
            return Err(TyperError::message(format!(
                "{caller}: subclassrange_max not Signed, got {max_val:?}"
            )));
        };
        // Carry the range markers as inheritance ids; the id is resolved
        // eagerly at rtype time (read off the vtable here) and `value` is
        // emitted verbatim.  The owning classdef is not in scope (only the
        // vtable `_ptr`), so `cdef_id` is `None`.
        let c_min = Constant::with_concretetype(
            ConstValue::InheritanceId {
                cdef_id: None,
                is_max: false,
                value: min_n,
            },
            LowLevelType::Signed,
        );
        let c_max = Constant::with_concretetype(
            ConstValue::InheritanceId {
                cdef_id: None,
                is_max: true,
                value: max_n,
            },
            LowLevelType::Signed,
        );
        // upstream: `gendirectcall(ll_issubclass_const, v_cls1,
        //   minid, maxid)`.
        let helper = rtyper.lowlevel_helper_function(
            "ll_issubclass_const",
            vec![
                CLASSTYPE.clone(),
                LowLevelType::Signed,
                LowLevelType::Signed,
            ],
            LowLevelType::Bool,
        )?;
        return hop.gendirectcall(
            &helper,
            vec![v_cls1, Hlvalue::Constant(c_min), Hlvalue::Constant(c_max)],
        );
    }

    // upstream variable case: `gendirectcall(ll_issubclass, v_cls1, v_cls2)`.
    let helper = rtyper.lowlevel_helper_function(
        "ll_issubclass",
        vec![CLASSTYPE.clone(), CLASSTYPE.clone()],
        LowLevelType::Bool,
    )?;
    hop.gendirectcall(&helper, vec![v_cls1, v_cls2])
}

/// RPython `cast_vtable_to_typeptr(vtable)` (rclass.py).
///
/// ```python
/// def cast_vtable_to_typeptr(vtable):
///     while typeOf(vtable).TO != OBJECT_VTABLE:
///         vtable = vtable.super
///     return vtable
/// ```
///
/// Walks the per-class vtable Struct's `super` chain inward until the
/// pointee type matches `OBJECT_VTABLE` (the root vtable Struct). At
/// the OBJECT_VTABLE level the upstream `vtable.super` access would
/// raise `AttributeError`, so the walk terminates by struct-type
/// equality before that point.
pub fn cast_vtable_to_typeptr(mut vtable: _ptr) -> _ptr {
    let object_vtable = OBJECT_VTABLE.clone();
    loop {
        let to_lltype = LowLevelType::from(vtable._TYPE.TO.clone());
        if to_lltype == object_vtable {
            return vtable;
        }
        // upstream: `vtable = vtable.super`. The Rust port follows
        // `_ptr.getattr("super")` which yields LowLevelValue::Ptr for
        // raw substructs (per `_ptr._expose`).
        let super_lv = vtable
            .getattr("super")
            .expect("vtable.super: per-class vtable must have a 'super' field");
        let crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(super_ptr) =
            super_lv
        else {
            panic!(
                "cast_vtable_to_typeptr: vtable.super did not yield a Ptr value (got {:?})",
                super_lv
            );
        };
        vtable = *super_ptr;
    }
}

impl Repr for ClassRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ClassRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// `RPythonTyper.translate_op_issubtype` (rtyper.py)
    /// dispatches `r.rtype_issubtype(hop)`. Without this override the
    /// trait default would surface `MissingRTypeOperation`. Forwards
    /// to the inherent [`ClassRepr::rtype_issubtype`] which mirrors
    /// upstream rclass.py.
    fn rtype_issubtype(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        ClassRepr::rtype_issubtype(self, hop)
    }

    /// RPython `ClassRepr.convert_desc(self, desc)` (rclass.py).
    ///
    /// ```python
    /// def convert_desc(self, desc):
    ///     subclassdef = desc.getuniqueclassdef()
    ///     if self.classdef is not None:
    ///         if self.classdef.commonbase(subclassdef) != self.classdef:
    ///             raise TyperError("not a subclass of %r: %r" % (
    ///                 self.classdef.name, desc))
    ///     r_subclass = getclassrepr(self.rtyper, subclassdef)
    ///     return r_subclass.getruntime(self.lowleveltype)
    /// ```
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        let crate::annotator::description::DescEntry::Class(class_rc) = desc else {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_desc: expected ClassDesc, got {desc:?}"
            )));
        };
        let subclassdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        // upstream `self.classdef.commonbase(subclassdef) != self.classdef`
        // — verify the candidate is a subclass of the receiver classdef.
        let common =
            crate::annotator::classdesc::ClassDef::commonbase(&self.classdef, &subclassdef);
        let same_class = match &common {
            Some(c) => Rc::ptr_eq(c, &self.classdef),
            None => false,
        };
        if !same_class {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_desc: {:?} is not a subclass of {:?}",
                subclassdef.borrow().name,
                self.classdef.borrow().name
            )));
        }
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.convert_desc: rtyper weak ref expired")
        })?;
        let r_subclass = getclassrepr_arc(&rtyper, Some(&subclassdef))?;
        let vtable_ptr = r_subclass.getruntime(&self.lowleveltype)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            self.lowleveltype.clone(),
        ))
    }

    /// RPython `ClassRepr.convert_const(self, value)` (rclass.py).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, (type, types.ClassType)):
    ///         raise TyperError("not a class: %r" % (value,))
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     return self.convert_desc(bk.getdesc(value))
    /// ```
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_const: not a class: {value:?}"
            )));
        };
        if !host.is_class() {
            return Err(TyperError::message(format!(
                "ClassRepr.convert_const: not a class: {host:?}"
            )));
        }
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.convert_const: rtyper weak ref expired")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }

    /// RPython `ClassRepr._setup_repr` (rclass.py).
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr._setup_repr: RPythonTyper weak ref expired")
        })?;
        let (name, basedef) = {
            let cd = self.classdef.borrow();
            (cd.name.clone(), cd.basedef.clone())
        };

        // rclass.py:247-249 — `clsfields = {}; pbcfields = {};
        // allmethods = {}`. Populate local copies first (upstream's
        // "don't store mutable objects on self before they are fully
        // built" rule at rclass.py:243-245).
        let mut clsfields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let mut pbcfields: HashMap<(usize, String), (String, Arc<dyn Repr>)> = HashMap::new();
        let mut allmethods: HashMap<String, bool> = HashMap::new();
        let mut llfields: Vec<(String, LowLevelType)> = Vec::new();

        // rclass.py:252-262 — attrs iteration with readonly filter.
        let attrs_sorted = {
            let cd = self.classdef.borrow();
            cd.attrs
                .iter()
                .map(|(k, v)| (k.clone(), (v.s_value.clone(), v.readonly)))
                .collect::<BTreeMap<String, (SomeValue, bool)>>()
        };
        for (attr_name, (s_value, readonly)) in &attrs_sorted {
            if !readonly {
                continue;
            }
            let mut s_value = s_value.clone();
            if let Some(s_unboundmethod) = self.prepare_method(&s_value)? {
                allmethods.insert(attr_name.clone(), true);
                s_value = s_unboundmethod;
            }
            let r = rtyper.getrepr(&s_value)?;
            let mangled_name = format!("cls_{attr_name}");
            clsfields.insert(attr_name.clone(), (mangled_name.clone(), r.clone()));
            llfields.push((mangled_name, r.lowleveltype().clone()));
        }
        for (access_set, attr_name, counter) in self.classdef.borrow().extra_access_sets.values() {
            let r = rtyper.getrepr(&access_set.borrow().s_value)?;
            let mangled_name = mangle(&format!("pbc{counter}"), attr_name);
            pbcfields.insert(
                (class_attr_family_key(access_set), attr_name.clone()),
                (mangled_name.clone(), r.clone()),
            );
            llfields.push((mangled_name, r.lowleveltype().clone()));
        }
        sort_llfields(&mut llfields);

        // rclass.py — `self.rbase = getclassrepr(self.rtyper,
        // self.classdef.basedef)`.
        let rbase = getclassrepr_arc(&rtyper, basedef.as_ref())?;
        // rclass.py:274 — `self.rbase.setup()`.
        Repr::setup(rbase.as_repr().as_ref())?;

        // rclass.py:275-278 — `kwds = {'hints': {...}}; vtable_type =
        // Struct('%s_vtable' % name, ('super', rbase.vtable_type),
        // *llfields, **kwds)`.
        let super_field_type = rbase.vtable_type().clone();
        let mut fields = Vec::with_capacity(1 + llfields.len());
        fields.push(("super".into(), super_field_type));
        fields.extend(llfields);
        let vtable_body = Struct::with_hints(
            &format!("{name}_vtable"),
            fields,
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        // rclass.py — `self.vtable_type.become(vtable_type)`. The
        // inner `Arc<Mutex<_>>` propagates to `self.lowleveltype`'s Ptr
        // target via the clone in `new()`.
        let LowLevelType::ForwardReference(fwd) = &self.vtable_type else {
            return Err(TyperError::message(
                "ClassRepr.vtable_type must be LowLevelType::ForwardReference",
            ));
        };
        fwd.r#become(LowLevelType::Struct(Box::new(vtable_body)))
            .map_err(TyperError::message)?;

        // rclass.py — `allmethods.update(self.rbase.allmethods)`.
        // Merge whichever parent methods upstream already recorded.
        if let ClassReprArc::Inst(parent_inst) = &rbase {
            for (method_name, _) in parent_inst.allmethods.borrow().iter() {
                allmethods.insert(method_name.clone(), true);
            }
        }

        // rclass.py:281-284 — publish the populated dicts onto `self`.
        *self.clsfields.borrow_mut() = clsfields;
        *self.pbcfields.borrow_mut() = pbcfields;
        *self.allmethods.borrow_mut() = allmethods;
        *self.rbase.borrow_mut() = Some(rbase);
        // `self.vtable = None` is represented by the absence of any R3
        // vtable slot on this struct.
        Ok(())
    }
}

/// Typed return for [`getclassrepr_arc`]: the internal recursion in
/// [`ClassRepr::_setup_repr`] needs direct access to `rbase.vtable_type`
/// and the Repr `setup()` entry-point, both of which are awkward to
/// reach through the public `Arc<dyn Repr>`. The enum branches mirror
/// upstream's class hierarchy (`ClassRepr` vs its `RootClassRepr`
/// subclass at rclass.py) without requiring runtime type reflection.
#[derive(Clone, Debug)]
pub(crate) enum ClassReprArc {
    Root(Arc<RootClassRepr>),
    Inst(Arc<ClassRepr>),
}

impl ClassReprArc {
    /// RPython `rbase.vtable_type` reader (rclass.py). Returns the
    /// `LowLevelType` that `ClassRepr._setup_repr` plugs into the
    /// `('super', ...)` field of the child vtable Struct.
    pub fn vtable_type(&self) -> &LowLevelType {
        match self {
            Self::Root(r) => r.vtable_type(),
            Self::Inst(r) => r.vtable_type(),
        }
    }

    /// Upcast to `Arc<dyn Repr>` so callers can invoke `Repr::setup` or
    /// hand the value to code that only needs the Repr surface (e.g.
    /// [`ExceptionData::r_exception_type`]).
    pub fn as_repr(&self) -> Arc<dyn Repr> {
        match self {
            Self::Root(r) => r.clone() as Arc<dyn Repr>,
            Self::Inst(r) => r.clone() as Arc<dyn Repr>,
        }
    }

    /// RPython `ClassRepr.getruntime(self, expected_type)` polymorphic
    /// entry. Both branches assert `expected_type == CLASSTYPE` and
    /// return `getvtable()`.
    pub fn getruntime(&self, expected_type: &LowLevelType) -> Result<_ptr, TyperError> {
        match self {
            Self::Root(r) => r.getruntime(expected_type),
            Self::Inst(r) => r.getruntime(expected_type),
        }
    }
}

// rclass.py — RootClassRepr (classdef = None flavour).

/// RPython `class RootClassRepr(ClassRepr)` (rclass.py).
///
/// Upstream inherits all `ClassRepr` methods and overrides `__init__` /
/// `_setup_repr` / `init_vtable`. The Rust port keeps `RootClassRepr` as
/// a sibling struct sharing the [`ClassReprArc`] dispatch enum because
/// Rust has no structural inheritance. [`RootClassRepr::_setup_repr`] is
/// a no-op (clsfields/pbcfields/allmethods/vtable all start empty) until
/// R2-B / R3 add their storage.
#[derive(Debug)]
pub struct RootClassRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:425). Weak backref —
    /// `init_vtable` calls `getinstancerepr(rtyper, None)` for the
    /// `rtti` slot of the root vtable.
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef = None` (rclass.py:422).
    classdef: Option<ClassDefKey>,
    /// RPython `self.vtable_type = OBJECT_VTABLE` (rclass.py).
    vtable_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.vtable_type)` (rclass.py:427).
    lowleveltype: LowLevelType,
    /// RPython `self.vtable = None` (rclass.py:433). Lazily filled by
    /// [`Self::init_vtable`].
    vtable: RefCell<Option<_ptr>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl RootClassRepr {
    /// RPython `RootClassRepr.__init__(self, rtyper)` (rclass.py).
    pub fn new(rtyper: &Rc<RPythonTyper>) -> Self {
        RootClassRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: None,
            vtable_type: OBJECT_VTABLE.clone(),
            lowleveltype: CLASSTYPE.clone(),
            vtable: RefCell::new(None),
            state: ReprState::new(),
        }
    }

    /// RPython `RootClassRepr.classdef` (`classdef = None`, rclass.py).
    pub fn classdef(&self) -> Option<ClassDefKey> {
        self.classdef
    }

    /// RPython `RootClassRepr.vtable_type` (rclass.py).
    pub fn vtable_type(&self) -> &LowLevelType {
        &self.vtable_type
    }

    /// RPython `RootClassRepr.init_vtable(self)` (rclass.py).
    ///
    /// ```python
    /// def init_vtable(self):
    ///     self.vtable = malloc(self.vtable_type, immortal=True)
    ///     self.fill_vtable_root(self.vtable)
    /// ```
    ///
    /// The root has no `super` chain, so allocation + a single
    /// `fill_vtable_root` finalises the OBJECT_VTABLE slot. Idempotent
    /// via the `vtable: Option<_ptr>` cache.
    pub fn init_vtable(&self) -> Result<(), TyperError> {
        // upstream malloc rejects ForwardReference, so resolve the
        // OBJECT_VTABLE forward-ref before passing.
        let vtable_lltype = match self.vtable_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "RootClassRepr.init_vtable: OBJECT_VTABLE ForwardReference not resolved",
                )
            })?,
            other => other,
        };
        let mut vtable = crate::translator::rtyper::lltypesystem::lltype::malloc(
            vtable_lltype,
            None,
            crate::translator::rtyper::lltypesystem::lltype::MallocFlavor::Gc,
            true,
        )
        .map_err(TyperError::message)?;
        self.fill_vtable_root(&mut vtable)?;
        *self.vtable.borrow_mut() = Some(vtable);
        Ok(())
    }

    /// RPython `ClassRepr.fill_vtable_root(self, vtable)` (rclass.py)
    /// adapted for the `classdef is None` arm.
    ///
    /// Upstream uses `0` / `sys.maxint` for the root subclassrange. The
    /// Rust port uses `i64::MAX` for the upper bound — RPython's
    /// `sys.maxint` is the platform `intptr_t.max`; on 64-bit hosts
    /// (the only target pyre supports today) that is `i64::MAX`. The
    /// `rtti` slot is populated via `getinstancerepr(rtyper, None)`
    /// + `getRuntimeTypeInfo(rinstance.object_type)`. The `name` slot
    ///   is the upstream `"object"` string. `classdef is None`, so
    ///   `hasattr(self.classdef, 'my_instantiate_graph')` is false and
    ///   the `instantiate` slot stays unset.
    pub fn fill_vtable_root(&self, vtable: &mut _ptr) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.fill_vtable_root: RPythonTyper weak ref expired")
        })?;
        // upstream: classdef is None → `subclassrange_min = 0,
        // subclassrange_max = sys.maxint`.
        vtable
            .setattr(
                "subclassrange_min",
                crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(0),
            )
            .map_err(TyperError::message)?;
        vtable
            .setattr(
                "subclassrange_max",
                crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Signed(i64::MAX),
            )
            .map_err(TyperError::message)?;
        // upstream: `rinstance = getinstancerepr(self.rtyper,
        // self.classdef); rinstance.setup(); if rinstance.gcflavor ==
        // 'gc': vtable.rtti = getRuntimeTypeInfo(rinstance.object_type)`.
        let rinstance = getinstancerepr(&rtyper, None, Flavor::Gc)?;
        Repr::setup(rinstance.as_ref() as &dyn Repr)?;
        if rinstance.gcflavor() == Flavor::Gc {
            let rtti = crate::translator::rtyper::lltypesystem::lltype::getRuntimeTypeInfo(
                rinstance.object_type(),
            )
            .map_err(TyperError::message)?;
            vtable
                .setattr(
                    "rtti",
                    crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(Box::new(
                        rtti,
                    )),
                )
                .map_err(TyperError::message)?;
        }
        let name_ptr = crate::translator::rtyper::lltypesystem::rstr::alloc_array_name("object")
            .map_err(TyperError::message)?;
        vtable
            .setattr(
                "name",
                crate::translator::rtyper::lltypesystem::lltype::LowLevelValue::Ptr(Box::new(
                    name_ptr,
                )),
            )
            .map_err(TyperError::message)?;
        Ok(())
    }

    /// RPython `ClassRepr.getvtable(self)` (rclass.py) for the
    /// rootclass.
    pub fn getvtable(&self) -> Result<_ptr, TyperError> {
        if self.vtable.borrow().is_none() {
            self.init_vtable()?;
        }
        let vtable = self
            .vtable
            .borrow()
            .clone()
            .expect("init_vtable post-condition: vtable is Some");
        // The root vtable IS the OBJECT_VTABLE level, so
        // `cast_vtable_to_typeptr` is a no-op (the `while typeOf(vtable).TO
        // != OBJECT_VTABLE` loop terminates immediately).
        Ok(cast_vtable_to_typeptr(vtable))
    }

    /// RPython `ClassRepr.getruntime(self, expected_type)` (rclass.py)
    /// for the rootclass.
    pub fn getruntime(&self, expected_type: &LowLevelType) -> Result<_ptr, TyperError> {
        if expected_type != &CLASSTYPE.clone() {
            return Err(TyperError::message(format!(
                "RootClassRepr.getruntime: expected CLASSTYPE, got {expected_type:?}"
            )));
        }
        self.getvtable()
    }

    /// RPython `ClassRepr.fromtypeptr(self, vcls, llops)` (rclass.py)
    /// for the rootclass — `lowleveltype` is `CLASSTYPE` and the cast
    /// reduces to a no-op-style `cast_pointer` that confirms the type.
    pub fn fromtypeptr(
        self: &Arc<Self>,
        vcls: Hlvalue,
        llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        Repr::setup(self.as_ref() as &dyn Repr)?;
        Ok(llops
            .genop(
                "cast_pointer",
                vec![vcls],
                GenopResult::LLType(self.lowleveltype.clone()),
            )
            .expect("cast_pointer with non-Void resulttype yields a Variable"))
    }

    /// RPython `ClassRepr.getclsfield` terminal arm (rclass.py):
    /// `raise MissingRTypeAttribute(attr)` when the rootclass has no
    /// matching clsfield. Pyre surfaces a structured `TyperError`.
    pub fn getclsfield(
        self: &Arc<Self>,
        _vcls: Hlvalue,
        attr: &str,
        _llops: &mut LowLevelOpList,
    ) -> Result<Variable, TyperError> {
        Err(TyperError::message(format!(
            "RootClassRepr.getclsfield: MissingRTypeAttribute({attr:?}) — \
             attribute not found in any clsfields"
        )))
    }

    /// RPython `ClassRepr.setclsfield` terminal arm (rclass.py):
    /// `raise MissingRTypeAttribute(attr)`.
    pub fn setclsfield(
        self: &Arc<Self>,
        _vcls: Hlvalue,
        attr: &str,
        _vvalue: Hlvalue,
        _llops: &mut LowLevelOpList,
    ) -> Result<(), TyperError> {
        Err(TyperError::message(format!(
            "RootClassRepr.setclsfield: MissingRTypeAttribute({attr:?}) — \
             attribute not found in any clsfields"
        )))
    }
}

impl Repr for RootClassRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        // Upstream `ClassRepr.__repr__` (rclass.py) emits
        // `"<ClassRepr for object>"`; the pyre tag follows the class name
        // matching the `.__class__.__name__` of RootClassRepr.
        "RootClassRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// `RPythonTyper.translate_op_issubtype` (rtyper.py)
    /// dispatches `r.rtype_issubtype(hop)`. Upstream's
    /// `RootClassRepr` inherits the body from `AbstractClassRepr`
    /// (rclass.py); the Rust port forwards through the shared
    /// [`rtype_issubtype_helper`] free fn so both ClassRepr and
    /// RootClassRepr surface the same op.
    fn rtype_issubtype(&self, hop: &crate::translator::rtyper::rtyper::HighLevelOp) -> RTypeResult {
        rtype_issubtype_helper(&self.rtyper, hop, "RootClassRepr.rtype_issubtype")
    }

    /// RPython `RootClassRepr._setup_repr(self)` (rclass.py).
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     self.clsfields = {}
    ///     self.pbcfields = {}
    ///     self.allmethods = {}
    ///     self.vtable = None
    /// ```
    ///
    /// All four assignments are currently no-ops because Phase R1 holds
    /// no storage for them — the fields only exist upstream to keep
    /// `ClassRepr` method dispatch consistent, and the methods that read
    /// them (`getvtable`, `getclsfield`, `get_field`) land with Phase R2.
    fn _setup_repr(&self) -> Result<(), TyperError> {
        Ok(())
    }

    /// RPython `ClassRepr.convert_desc(self, desc)` (rclass.py),
    /// inherited by `RootClassRepr`. With `classdef is None` the
    /// commonbase check is skipped (upstream `if self.classdef is not
    /// None: ...`).
    fn convert_desc(
        &self,
        desc: &crate::annotator::description::DescEntry,
    ) -> Result<Constant, TyperError> {
        let crate::annotator::description::DescEntry::Class(class_rc) = desc else {
            return Err(TyperError::message(format!(
                "RootClassRepr.convert_desc: expected ClassDesc, got {desc:?}"
            )));
        };
        let subclassdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc)
            .map_err(|e| TyperError::message(e.to_string()))?;
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.convert_desc: rtyper weak ref expired")
        })?;
        let r_subclass = getclassrepr_arc(&rtyper, Some(&subclassdef))?;
        let vtable_ptr = r_subclass.getruntime(&self.lowleveltype)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable_ptr)),
            self.lowleveltype.clone(),
        ))
    }

    /// RPython `ClassRepr.convert_const(self, value)` (rclass.py),
    /// inherited by `RootClassRepr`.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let ConstValue::HostObject(host) = value else {
            return Err(TyperError::message(format!(
                "RootClassRepr.convert_const: not a class: {value:?}"
            )));
        };
        if !host.is_class() {
            return Err(TyperError::message(format!(
                "RootClassRepr.convert_const: not a class: {host:?}"
            )));
        }
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.convert_const: rtyper weak ref expired")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("RootClassRepr.convert_const: annotator weak ref dropped")
        })?;
        let desc = annotator
            .bookkeeper
            .getdesc(host)
            .map_err(|e| TyperError::message(e.to_string()))?;
        self.convert_desc(&desc)
    }
}

// rclass.py — InstanceRepr (classdef=None + classdef!=None
// scaffolding branches).

/// RPython `class InstanceRepr(Repr)` (rclass.py).
///
/// R2-C brings in the attrs iteration for `readonly = False` instance
/// attributes: `fields` / `allinstancefields` populate via
/// `rtyper.getrepr(attrdef.s_value)` + `mangled_name = 'inst_' + name`
/// (rclass.py), and the `object_type` Struct body includes the
/// `(mangled_name, r.lowleveltype)` pairs alongside `('super',
/// rbase.object_type)`. The `special_memory_pressure` /
/// `mutate_<name>` quasi-immutable fields (rclass.py), the
/// `_check_for_immutable_hints` branch (rclass.py), and the
/// `rtti=True` kwarg (rclass.py `attachRuntimeTypeInfo` side
/// effect) remain R2-D / R3 scope.
#[derive(Debug)]
pub struct InstanceRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:469).
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef` (rclass.py:470). `None` mirrors
    /// upstream's sentinel for the root instance that carries
    /// `object_type = OBJECT_BY_FLAVOR[...]` directly.
    classdef: Option<Rc<RefCell<ClassDef>>>,
    /// RPython `self.object_type = OBJECT_BY_FLAVOR[LLFLAVOR[gcflavor]]`
    /// when classdef is None (rclass.py), otherwise a fresh
    /// `ForwardReference` (rclass.py). The FR target resolves
    /// during `_setup_repr` once the `MkStruct('name', ('super',
    /// rbase.object_type), ...)` body is built (rclass.py).
    object_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.object_type)` (rclass.py).
    lowleveltype: LowLevelType,
    /// RPython `self.gcflavor` (rclass.py:478).
    gcflavor: Flavor,
    /// RPython `self.rclass = getclassrepr(...)` (rclass.py).
    rclass: RefCell<Option<ClassReprArc>>,
    /// RPython `self.fields = fields` (rclass.py:557). Instance-level
    /// attributes keyed by unmangled attr name; populated by
    /// `_setup_repr`.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    fields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.rbase = getinstancerepr(...)` (rclass.py).
    rbase: RefCell<Option<Arc<InstanceRepr>>>,
    /// RPython `self.allinstancefields = allinstancefields`
    /// (rclass.py:558). Accumulates rbase.allinstancefields ∪ fields.
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    allinstancefields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.immutable_field_set = set()` (rclass.py) for
    /// classdef=None, overwritten by `_check_for_immutable_hints` for
    /// classdef!=None (rclass.py). R2-C keeps it as an empty set —
    /// the immutable-hint derivation lands in R2-D.
    #[allow(dead_code)]
    immutable_field_set: RefCell<HashSet<String>>,
    /// RPython `self._reusable_prebuilt_instance` (rclass.py:807).
    /// Lazy-initialised by [`Self::get_reusable_prebuilt_instance`] —
    /// stays `None` until the first call.
    reusable_prebuilt_instance: RefCell<Option<_ptr>>,
    /// RPython `self.iprebuiltinstances = identity_dict()` (rclass.py).
    /// Value-keyed identity cache populated by
    /// [`Self::convert_const_exact`]. The HostObject is the upstream
    /// `value` argument; `Hash`/`Eq` use Arc identity matching
    /// upstream's `identity_dict`. Stays empty until the first
    /// exact-match `convert_const` reaches the cache.
    iprebuiltinstances: RefCell<HashMap<HostObject, _ptr>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
    /// Extra state of `rvirtualizable.py VirtualizableInstanceRepr`.
    /// `None` on a plain `InstanceRepr`.
    virtualizable: RefCell<Option<super::rvirtualizable::VirtualizableInstanceRepr>>,
}

impl InstanceRepr {
    /// RPython `InstanceRepr.__init__(self, rtyper, classdef,
    /// gcflavor='gc')` (rclass.py).
    ///
    /// Covers both branches: `classdef is None` (rclass.py) and
    /// `classdef is not None` (rclass.py). In the latter case a
    /// fresh `ForwardReference` / `GcForwardReference` is allocated with
    /// its `Arc<Mutex<_>>` target shared into the `lowleveltype` Ptr, so
    /// [`InstanceRepr::_setup_repr`] can resolve it once.
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        gcflavor: Flavor,
    ) -> Self {
        let (object_type, lowleveltype) = build_instance_types(classdef, gcflavor);
        InstanceRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: classdef.cloned(),
            object_type,
            lowleveltype,
            gcflavor,
            rclass: RefCell::new(None),
            fields: RefCell::new(HashMap::new()),
            rbase: RefCell::new(None),
            allinstancefields: RefCell::new(HashMap::new()),
            immutable_field_set: RefCell::new(HashSet::new()),
            reusable_prebuilt_instance: RefCell::new(None),
            iprebuiltinstances: RefCell::new(HashMap::new()),
            state: ReprState::new(),
            virtualizable: RefCell::new(None),
        }
    }

    /// Convenience constructor for the classdef=None root instance
    /// (rclass.py:471-472). Keeps the live `rtyper` backref so
    /// `_setup_repr` can populate the root `__class__` field exactly
    /// like upstream.
    pub fn new_rootinstance(rtyper: &Rc<RPythonTyper>, gcflavor: Flavor) -> Self {
        Self::new(rtyper, None, gcflavor)
    }

    /// RPython `InstanceRepr.classdef` (rclass.py). Returns
    /// `None` for root instances, `Some(Rc<RefCell<ClassDef>>)` otherwise.
    pub fn classdef(&self) -> Option<Rc<RefCell<ClassDef>>> {
        self.classdef.clone()
    }

    /// RPython `InstanceRepr.object_type` (rclass.py).
    pub fn object_type(&self) -> &LowLevelType {
        &self.object_type
    }

    /// RPython `InstanceRepr.gcflavor` (rclass.py).
    pub fn gcflavor(&self) -> Flavor {
        self.gcflavor
    }

    /// RPython `InstanceRepr.common_repr(self)` (rclass.py):
    ///
    /// ```python
    /// def common_repr(self):  # -> object or nongcobject reprs
    ///     return getinstancerepr(self.rtyper, None, self.gcflavor)
    /// ```
    ///
    /// Returns the root `InstanceRepr` for this instance's gcflavor —
    /// the one whose `lowleveltype` is `OBJECTPTR` (or `NONGCOBJECTPTR`
    /// for non-GC flavors). `rtype_isinstance` (rclass.py)
    /// uses this to convert `v_obj` to the common base type before the
    /// runtime check.
    pub fn common_repr(&self) -> Result<Arc<InstanceRepr>, TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.common_repr: rtyper weak ref expired")
        })?;
        getinstancerepr(&rtyper, None, self.gcflavor)
    }

    /// Read-only view of the instance-level `fields` dict populated by
    /// `_setup_repr`. RPython `self.fields` (rclass.py).
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub fn fields(&self) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.fields.borrow()
    }

    /// Read-only view of the `allinstancefields` dict. RPython
    /// `self.allinstancefields` (rclass.py:558).
    #[expect(
        clippy::type_complexity,
        reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
    )]
    pub fn allinstancefields(
        &self,
    ) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.allinstancefields.borrow()
    }

    /// RPython `InstanceRepr.rclass` (rclass.py) — the
    /// [`ClassRepr`] / [`RootClassRepr`] handle for this instance's
    /// class. Populated lazily by [`InstanceRepr::_setup_repr`]; callers
    /// must `Repr::setup` first or be invoked from a context where
    /// setup has run.
    pub fn rclass(&self) -> Option<ClassReprArc> {
        self.rclass.borrow().clone()
    }

    /// RPython `InstanceRepr.getfieldrepr(self, attr)` (rclass.py):
    ///
    /// ```python
    /// def getfieldrepr(self, attr):
    ///     """Return the repr used for the given attribute."""
    ///     if attr in self.fields:
    ///         mangled_name, r = self.fields[attr]
    ///         return r
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         return self.rbase.getfieldrepr(attr)
    /// ```
    pub fn getfieldrepr(self: &Arc<Self>, attr: &str) -> Result<Arc<dyn Repr>, TyperError> {
        if let Some((_mangled, r)) = self.fields.borrow().get(attr).cloned() {
            return Ok(r);
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        if self.classdef.is_none() {
            return Err(TyperError::message(format!(
                "InstanceRepr.getfieldrepr: MissingRTypeAttribute({attr:?})"
            )));
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.getfieldrepr: rbase missing — call setup() first")
        })?;
        rbase.getfieldrepr(attr).map_err(|err| {
            // Keep the failing owner's annotation beside its built fields:
            // this distinguishes an absent attribute from premature repr
            // setup without changing rclass.py::InstanceRepr.getfieldrepr's
            // lookup or allowing a missing field.
            let classdef = self.classdef.as_ref().unwrap().borrow();
            let attribute = classdef.attrs.get(attr).map(|value| {
                (value.readonly, value.s_value.clone())
            });
            let mut fields: Vec<_> = self.fields.borrow().keys().cloned().collect();
            fields.sort();
            TyperError::message(format!(
                "{err}; InstanceRepr {:?}: attribute {attr:?} (readonly, value)={attribute:?}, built fields={fields:?}",
                classdef.name
            ))
        })
    }

    /// RPython `InstanceRepr.hook_access_field(self, vinst, cname,
    /// llops, flags)` (rclass.py):
    ///
    /// ```python
    /// def hook_access_field(self, vinst, cname, llops, flags):
    ///     pass # for virtualizables; see rvirtualizable.py
    /// ```
    ///
    /// Default no-op. Subclasses (e.g. virtualizable instance reprs)
    /// override to emit `promote_virtualizable` op before the getfield.
    pub fn hook_access_field(
        &self,
        vinst: &Hlvalue,
        cname: &Hlvalue,
        llops: &mut LowLevelOpList,
        flags: &Flags,
    ) {
        let Some(vable) = self.virtualizable.borrow().clone() else {
            return;
        };
        let Some(name) = hlvalue_byte_str(cname) else {
            return;
        };
        if !vable.should_force_field(&name) {
            return;
        }
        // `cflags = inputconst(lltype.Void, flags)`
        // `llops.genop('jit_force_virtualizable', [vinst, cname, cflags])`
        let mut items = HashMap::new();
        for (key, value) in flags {
            items.insert(ConstValue::byte_str(key), value.clone());
        }
        let cflags = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::Dict(items),
            LowLevelType::Void,
        ));
        llops.genop(
            "jit_force_virtualizable",
            vec![vinst.clone(), cname.clone(), cflags],
            GenopResult::Void,
        );
    }

    /// RPython `InstanceRepr.hook_setfield(self, vinst, fieldname,
    /// llops)` (rclass.py):
    ///
    /// ```python
    /// def hook_setfield(self, vinst, fieldname, llops):
    ///     if self.is_quasi_immutable(fieldname):
    ///         c_fieldname = inputconst(Void, 'mutate_' + fieldname)
    ///         llops.genop('jit_force_quasi_immutable', [vinst, c_fieldname])
    /// ```
    pub fn hook_setfield(
        self: &Arc<Self>,
        vinst: &Hlvalue,
        fieldname: &str,
        llops: &mut LowLevelOpList,
    ) {
        if self.is_quasi_immutable(fieldname) {
            let c_fieldname = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(format!("mutate_{fieldname}")),
                LowLevelType::Void,
            ));
            llops.genop(
                "jit_force_quasi_immutable",
                vec![vinst.clone(), c_fieldname],
                GenopResult::Void,
            );
        }
    }

    /// RPython `InstanceRepr.is_quasi_immutable(self, fieldname)`
    /// (rclass.py):
    ///
    /// ```python
    /// def is_quasi_immutable(self, fieldname):
    ///     search1 = fieldname + '?'
    ///     search2 = fieldname + '?[*]'
    ///     rbase = self
    ///     while rbase.classdef is not None:
    ///         if (search1 in rbase.immutable_field_set or
    ///                 search2 in rbase.immutable_field_set):
    ///             return True
    ///         rbase = rbase.rbase
    ///     return False
    /// ```
    pub fn is_quasi_immutable(self: &Arc<Self>, fieldname: &str) -> bool {
        let search1 = format!("{fieldname}?");
        let search2 = format!("{fieldname}?[*]");
        let mut rbase = self.clone();
        while rbase.classdef.is_some() {
            {
                let set = rbase.immutable_field_set.borrow();
                if set.contains(&search1) || set.contains(&search2) {
                    return true;
                }
            }
            let next = rbase.rbase.borrow().clone();
            match next {
                Some(n) => rbase = n,
                None => break,
            }
        }
        false
    }

    /// RPython `InstanceRepr.getfield(self, vinst, attr, llops,
    /// force_cast=False, flags={})` (rclass.py):
    ///
    /// ```python
    /// def getfield(self, vinst, attr, llops, force_cast=False, flags={}):
    ///     """Read the given attribute (or __class__ for the type) of 'vinst'."""
    ///     if attr in self.fields:
    ///         mangled_name, r = self.fields[attr]
    ///         cname = inputconst(Void, mangled_name)
    ///         if force_cast:
    ///             vinst = llops.genop('cast_pointer', [vinst], resulttype=self)
    ///         self.hook_access_field(vinst, cname, llops, flags)
    ///         return llops.genop('getfield', [vinst, cname], resulttype=r)
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         return self.rbase.getfield(vinst, attr, llops, force_cast=True,
    ///                                    flags=flags)
    /// ```
    pub fn getfield(
        self: &Arc<Self>,
        vinst: Hlvalue,
        attr: &str,
        llops: &mut LowLevelOpList,
        force_cast: bool,
        flags: &Flags,
    ) -> Result<Variable, TyperError> {
        if let Some((mangled_name, r)) = self.fields.borrow().get(attr).cloned() {
            // upstream: `cname = inputconst(Void, mangled_name)`.
            let cname = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(mangled_name.clone()),
                LowLevelType::Void,
            ));
            // upstream: `if force_cast: vinst = llops.genop('cast_pointer',
            //                                              [vinst], resulttype=self)`.
            let vinst = if force_cast {
                let v_cast = llops
                    .genop(
                        "cast_pointer",
                        vec![vinst],
                        GenopResult::LLType(self.lowleveltype.clone()),
                    )
                    .expect("cast_pointer with non-Void resulttype yields a Variable");
                Hlvalue::Variable(v_cast)
            } else {
                vinst
            };
            // upstream: `self.hook_access_field(vinst, cname, llops, flags)`.
            self.hook_access_field(&vinst, &cname, llops, flags);
            // upstream: `return llops.genop('getfield', [vinst, cname],
            //                              resulttype=r)`.
            return Ok(llops
                .genop(
                    "getfield",
                    vec![vinst, cname],
                    GenopResult::LLType(r.lowleveltype().clone()),
                )
                .expect("getfield with non-Void resulttype yields a Variable"));
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        if self.classdef.is_none() {
            return Err(TyperError::message(format!(
                "InstanceRepr.getfield: MissingRTypeAttribute({attr:?})"
            )));
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.getfield: rbase missing — call setup() first")
        })?;
        // upstream: `return self.rbase.getfield(vinst, attr, llops,
        //                                      force_cast=True, flags=flags)`.
        rbase.getfield(vinst, attr, llops, true, flags)
    }

    /// RPython `InstanceRepr.setfield(self, vinst, attr, vvalue, llops,
    /// force_cast=False, flags={})` (rclass.py):
    ///
    /// ```python
    /// def setfield(self, vinst, attr, vvalue, llops, force_cast=False, flags={}):
    ///     """Write the given attribute (or __class__ for the type) of 'vinst'."""
    ///     if attr in self.fields:
    ///         mangled_name, r = self.fields[attr]
    ///         cname = inputconst(Void, mangled_name)
    ///         if force_cast:
    ///             vinst = llops.genop('cast_pointer', [vinst], resulttype=self)
    ///         self.hook_access_field(vinst, cname, llops, flags)
    ///         llops.genop('setfield', [vinst, cname, vvalue])
    ///     else:
    ///         if self.classdef is None:
    ///             raise MissingRTypeAttribute(attr)
    ///         self.rbase.setfield(vinst, attr, vvalue, llops, force_cast=True,
    ///                             flags=flags)
    /// ```
    ///
    /// Mirror of [`Self::getfield`] for the write side. `__class__`
    /// resolves through the `rbase` fall-through to the root instance's
    /// `typeptr` slot (`force_cast=True` on each step).
    pub fn setfield(
        self: &Arc<Self>,
        vinst: Hlvalue,
        attr: &str,
        vvalue: Hlvalue,
        llops: &mut LowLevelOpList,
        force_cast: bool,
        flags: &Flags,
    ) -> Result<(), TyperError> {
        if let Some((mangled_name, _r)) = self.fields.borrow().get(attr).cloned() {
            // upstream: `cname = inputconst(Void, mangled_name)`.
            let cname = Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::byte_str(mangled_name.clone()),
                LowLevelType::Void,
            ));
            // upstream: `if force_cast: vinst = llops.genop('cast_pointer',
            //                                              [vinst], resulttype=self)`.
            let vinst = if force_cast {
                let v_cast = llops
                    .genop(
                        "cast_pointer",
                        vec![vinst],
                        GenopResult::LLType(self.lowleveltype.clone()),
                    )
                    .expect("cast_pointer with non-Void resulttype yields a Variable");
                Hlvalue::Variable(v_cast)
            } else {
                vinst
            };
            // upstream: `self.hook_access_field(vinst, cname, llops, flags)`.
            self.hook_access_field(&vinst, &cname, llops, flags);
            // upstream: `self.hook_setfield(vinst, attr, llops)`.
            self.hook_setfield(&vinst, attr, llops);
            // upstream: `llops.genop('setfield', [vinst, cname, vvalue])`.
            llops.genop("setfield", vec![vinst, cname, vvalue], GenopResult::Void);
            return Ok(());
        }
        // upstream: `if self.classdef is None: raise MissingRTypeAttribute(attr)`.
        if self.classdef.is_none() {
            return Err(TyperError::message(format!(
                "InstanceRepr.setfield: MissingRTypeAttribute({attr:?})"
            )));
        }
        let rbase = self.rbase.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.setfield: rbase missing — call setup() first")
        })?;
        // upstream: `self.rbase.setfield(vinst, attr, vvalue, llops,
        //                               force_cast=True, flags=flags)`.
        rbase.setfield(vinst, attr, vvalue, llops, true, flags)
    }

    /// RPython `InstanceRepr.new_instance(self, llops,
    /// classcallhop=None, nonmovable=False)` (rclass.py):
    ///
    /// ```python
    /// def new_instance(self, llops, classcallhop=None, nonmovable=False):
    ///     """Build a new instance, without calling __init__."""
    ///     flavor = self.gcflavor
    ///     flags = {'flavor': flavor}
    ///     if nonmovable:
    ///         flags['nonmovable'] = True
    ///     ctype = inputconst(Void, self.object_type)
    ///     cflags = inputconst(Void, flags)
    ///     vlist = [ctype, cflags]
    ///     vptr = llops.genop('malloc', vlist, resulttype=Ptr(self.object_type))
    ///     ctypeptr = inputconst(CLASSTYPE, self.rclass.getvtable())
    ///     self.setfield(vptr, '__class__', ctypeptr, llops)
    ///     if self.has_special_memory_pressure(self.object_type):
    ///         self.setfield(vptr, 'special_memory_pressure',
    ///             inputconst(lltype.Signed, 0), llops)
    ///     ...                # initialize instance attributes from defaults
    ///     return vptr
    /// ```
    ///
    /// Emits the runtime `malloc` op (vs the host-side immortal
    /// allocation [`Self::create_instance`] used for prebuilt data) plus
    /// the `__class__` `setfield` and the class-level defaults loop.
    /// `classcallhop` is unused in this base body, exactly as upstream —
    /// its only consumer is the `TaggedInstanceRepr` override
    /// (`lltypesystem/rtagged.py`), unported.
    pub fn new_instance(
        self: &Arc<Self>,
        llops: &mut LowLevelOpList,
        _classcallhop: Option<&crate::translator::rtyper::rtyper::HighLevelOp>,
        nonmovable: bool,
    ) -> Result<Hlvalue, TyperError> {
        // upstream: `ctype = inputconst(Void, self.object_type)` — the
        // Void-typed type tag carrying the resolved object Struct lltype,
        // exactly as `TupleRepr::newtuple` encodes its malloc type arg.
        let object_lltype = match self.object_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.new_instance: object_type ForwardReference not \
                     resolved (call setup() first)",
                )
            })?,
            other => other,
        };
        let ctype = Constant::with_concretetype(
            ConstValue::LowLevelType(Box::new(object_lltype.clone())),
            LowLevelType::Void,
        );
        // upstream: `cflags = inputconst(Void, {'flavor': flavor})` plus
        // `flags['nonmovable'] = True` when requested. The flavor
        // sentinel matches the `newtuple` emitter's encoding; the
        // lowering pass reads the type from `ctype` and the flavor here.
        let flavor_sentinel = match (self.gcflavor, nonmovable) {
            (Flavor::Gc, false) => "flavor=gc",
            (Flavor::Gc, true) => "flavor=gc;nonmovable",
            (Flavor::Raw, false) => "flavor=raw",
            (Flavor::Raw, true) => "flavor=raw;nonmovable",
        };
        let cflags =
            Constant::with_concretetype(ConstValue::byte_str(flavor_sentinel), LowLevelType::Void);
        // upstream: `vptr = llops.genop('malloc', vlist,
        //                              resulttype=Ptr(self.object_type))`.
        let vptr = llops
            .genop(
                "malloc",
                vec![Hlvalue::Constant(ctype), Hlvalue::Constant(cflags)],
                GenopResult::LLType(self.lowleveltype.clone()),
            )
            .expect("malloc with non-Void result yields a Variable");
        let vptr = Hlvalue::Variable(vptr);
        // upstream: `ctypeptr = inputconst(CLASSTYPE, self.rclass.getvtable())`.
        let rclass = self.rclass().ok_or_else(|| {
            TyperError::message(
                "InstanceRepr.new_instance: rclass not populated — call setup() first",
            )
        })?;
        // A ClassRepr freshly minted by `InstanceRepr._setup_repr` is
        // still on the rtyper's pending-setup list at this point
        // (`call_all_setups` only runs between block batches), and
        // `init_vtable` needs its `vtable_type` ForwardReference
        // resolved — set it up here, same guard style as
        // `rtype_new_instance` / `fromtypeptr`.  Idempotent.
        Repr::setup(rclass.as_repr().as_ref())?;
        // upstream `self.rclass.getvtable()` — `getruntime(CLASSTYPE)` is
        // the `ClassReprArc`-polymorphic accessor that asserts the
        // expected type then returns the vtable pointer.
        let vtable = rclass.getruntime(&CLASSTYPE)?;
        let ctypeptr = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(vtable)),
            CLASSTYPE.clone(),
        ));
        // upstream: `self.setfield(vptr, '__class__', ctypeptr, llops)`.
        self.setfield(
            vptr.clone(),
            "__class__",
            ctypeptr,
            llops,
            false,
            &Flags::default(),
        )?;
        // upstream rclass.py:745-748 — zero the
        // `special_memory_pressure` counter slot when the object struct
        // carries one.  The slot is minted by the
        // `_special_memory_pressure_` class hint (rclass.py:541-544,
        // unported), so this arm stays dormant until that hint lands.
        if has_special_memory_pressure(&object_lltype) {
            self.setfield(
                vptr.clone(),
                "special_memory_pressure",
                Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::Int(0),
                    LowLevelType::Signed,
                )),
                llops,
                false,
                &Flags::default(),
            )?;
        }
        // upstream rclass.py — initialize instance attributes
        // from their class-level defaults:
        //
        //     if self.classdef is not None:
        //         flds = self.allinstancefields.keys()
        //         flds.sort()
        //         for fldname in flds:
        //             if fldname == '__class__':
        //                 continue
        //             mangled_name, r = self.allinstancefields[fldname]
        //             if r.lowleveltype is Void:
        //                 continue
        //             value = self.classdef.classdesc.read_attribute(fldname, None)
        //             if value is not None:
        //                 ll_value = r.convert_desc_or_const(value)
        //                 # don't write NULL GC pointers: we know that the malloc
        //                 # done above initialized at least the GC Ptr fields to
        //                 # NULL already, and that's true for all our GCs
        //                 if (isinstance(r.lowleveltype, Ptr) and
        //                         r.lowleveltype.TO._gckind == 'gc' and
        //                         not ll_value):
        //                     continue
        //                 cvalue = inputconst(r.lowleveltype, ll_value)
        //                 self.setfield(vptr, fldname, cvalue, llops,
        //                               flags={'access_directly': True})
        if let Some(classdef) = &self.classdef {
            let classdesc = classdef.borrow().classdesc.clone();
            // upstream `flds.sort()` — deterministic field order.
            #[expect(
                clippy::type_complexity,
                reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
            )]
            let mut flds: Vec<(String, (String, Arc<dyn Repr>))> = self
                .allinstancefields()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            flds.sort_by(|a, b| a.0.cmp(&b.0));
            for (fldname, (_mangled_name, r)) in &flds {
                if fldname == "__class__" {
                    continue;
                }
                if matches!(r.lowleveltype(), LowLevelType::Void) {
                    continue;
                }
                let Some(value) =
                    crate::annotator::classdesc::ClassDesc::read_attribute(&classdesc, fldname)
                else {
                    continue;
                };
                let entry = match value {
                    crate::annotator::classdesc::ClassDictEntry::Constant(c) => {
                        DescOrConst::Const(c)
                    }
                    crate::annotator::classdesc::ClassDictEntry::Desc(d) => DescOrConst::Desc(d),
                };
                let converted = r.convert_desc_or_const(&entry)?;
                // upstream: don't write NULL GC pointers — malloc already
                // zero-initialized the GC Ptr fields.
                if let LowLevelType::Ptr(p) = r.lowleveltype()
                    && p._gckind() == crate::translator::rtyper::lltypesystem::lltype::GcKind::Gc
                    && matches!(
                        &converted.value,
                        ConstValue::LLPtr(llp) if matches!(llp._obj0_value(), Ok(None))
                    )
                {
                    continue;
                }
                // upstream `cvalue = inputconst(r.lowleveltype, ll_value)` —
                // `convert_desc_or_const` already returned the Constant
                // with `r.lowleveltype` stamped.
                let mut flags = Flags::default();
                flags.insert("access_directly".to_string(), ConstValue::Bool(true));
                self.setfield(
                    vptr.clone(),
                    fldname,
                    Hlvalue::Constant(converted),
                    llops,
                    false,
                    &flags,
                )?;
            }
        }
        Ok(vptr)
    }

    /// RPython `InstanceRepr.null_instance(self)` (rclass.py):
    ///
    /// ```python
    /// def null_instance(self):
    ///     return nullptr(self.object_type)
    /// ```
    ///
    /// Returns a null `_ptr` to the same container as `self.lowleveltype`,
    /// suitable for embedding into `Constant(value=LLPtr(...),
    /// concretetype=self.lowleveltype)` (the upstream `convert_const`
    /// `value is None` branch).
    pub fn null_instance(&self) -> Result<_ptr, TyperError> {
        // rclass.InstanceRepr.null_instance passes object_type directly.
        // A null pointer needs no object layout; Ptr retains an unresolved
        // ForwardReference until ordinary repr setup resolves that identity.
        lltype::nullptr(self.object_type.clone()).map_err(TyperError::message)
    }

    /// RPython `InstanceRepr.upcast(self, result)` (rclass.py):
    ///
    /// ```python
    /// def upcast(self, result):
    ///     return cast_pointer(self.lowleveltype, result)
    /// ```
    ///
    /// Used by `convert_const` when delegating to a subclass `InstanceRepr`
    /// (rclass.py): the subclass produces a pointer typed at the
    /// subclass's `lowleveltype`; this helper casts it back up to the
    /// owning `InstanceRepr`'s `lowleveltype`.
    pub fn upcast(&self, result: &_ptr) -> Result<_ptr, TyperError> {
        let LowLevelType::Ptr(ptrtype) = &self.lowleveltype else {
            return Err(TyperError::message(format!(
                "InstanceRepr.upcast: self.lowleveltype is not a Ptr: {:?}",
                self.lowleveltype
            )));
        };
        lltype::cast_pointer(ptrtype, result).map_err(TyperError::message)
    }

    /// RPython `InstanceRepr.create_instance(self)` (rclass.py):
    ///
    /// ```python
    /// def create_instance(self):
    ///     return malloc(self.object_type, flavor=self.gcflavor, immortal=True)
    /// ```
    ///
    /// Allocates a fresh instance container in the appropriate gc flavor
    /// with `immortal=True` so it survives the prebuilt-data lifetime.
    /// The `object_type` ForwardReference (for non-root) must already be
    /// resolved by `_setup_repr`; the caller is responsible for
    /// `Repr::setup` first.
    pub fn create_instance(&self) -> Result<_ptr, TyperError> {
        let object_lltype = match self.object_type.clone() {
            LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.create_instance: object_type ForwardReference not \
                     resolved (call setup() first)",
                )
            })?,
            other => other,
        };
        let flavor = match self.gcflavor {
            Flavor::Gc => lltype::MallocFlavor::Gc,
            Flavor::Raw => lltype::MallocFlavor::Raw,
        };
        lltype::malloc(object_lltype, None, flavor, true).map_err(TyperError::message)
    }

    /// RPython `InstanceRepr.initialize_prebuilt_data(self, value,
    /// classdef, result)` (rclass.py).
    ///
    /// Recursive helper that walks the `rbase` chain (each level
    /// addressed via the per-step `path` of `"super"` segments,
    /// matching the `setattr_at_path` semantics introduced for vtable
    /// init) and writes each level's instance fields. Reaching the
    /// root (`classdef is None`) writes the `typeptr` slot to point at
    /// the leaf-class's vtable via `getclassrepr(rtyper, classdef)`.
    ///
    /// Pyre-port adaptations:
    /// - `value` is the live HostObject (or `None` for the upstream
    ///   `Ellipsis` sentinel meaning "use defaults"). The Ellipsis
    ///   path skips the live `instance_get` probe and falls through
    ///   straight to `read_attribute` → `_defl`.
    /// - Upstream's `getattr(value, name)` uses the host descriptor protocol;
    ///   only a missing attribute takes the class-declaration/default branch.
    pub fn initialize_prebuilt_data(
        &self,
        _value: Option<&HostObject>,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        result: &mut _ptr,
        path: &[&str],
    ) -> Result<(), TyperError> {
        if let Some(_self_classdef) = self.classdef.as_ref() {
            // upstream rclass.py:949 — recurse into rbase first.
            let rbase = self.rbase.borrow().clone().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.initialize_prebuilt_data: rbase missing — \
                     call setup() first",
                )
            })?;
            let mut next_path = Vec::with_capacity(path.len() + 1);
            next_path.extend_from_slice(path);
            next_path.push("super");
            rbase.initialize_prebuilt_data(_value, classdef, result, &next_path)?;
            // upstream rclass.py:951-971 — per-level fields:
            //   for name, (mangled_name, r) in self.fields.items():
            //       if r.lowleveltype is Void:
            //           llattrvalue = None
            //       else:
            //           try:
            //               attrvalue = getattr(value, name)
            //           except AttributeError:
            //               attrvalue = self.classdef.classdesc.read_attribute(
            //                   name, None)
            //               if attrvalue is None:
            //                   llattrvalue = r.lowleveltype._defl()
            //               else:
            //                   llattrvalue = r.convert_desc_or_const(attrvalue)
            //           else:
            //               llattrvalue = r.convert_const(attrvalue)
            //       setattr(result, mangled_name, llattrvalue)
            //
            // The Ellipsis-sentinel path
            // (`get_reusable_prebuilt_instance` with `_value = None`)
            // skips the host lookup entirely and goes
            // straight to `read_attribute` / `_defl`.
            #[expect(
                clippy::type_complexity,
                reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
            )]
            let fields_snapshot: Vec<(String, (String, Arc<dyn Repr>))> = self
                .fields
                .borrow()
                .iter()
                .map(|(name, (mangled, r))| (name.clone(), (mangled.clone(), r.clone())))
                .collect();
            if !fields_snapshot.is_empty() {
                let classdef_b = _self_classdef.borrow();
                let classdesc = classdef_b.classdesc.clone();
                drop(classdef_b);
                for (name, (mangled_name, r)) in &fields_snapshot {
                    let llattrvalue = if matches!(r.lowleveltype(), LowLevelType::Void) {
                        // upstream `if r.lowleveltype is Void: llattrvalue = None` —
                        // pyre's `_ptr.setattr` with a Void-tagged value
                        // is modeled as `LowLevelValue::Void`.
                        lltype::LowLevelValue::Void
                    } else {
                        // rclass.py:InstanceRepr.initialize_prebuilt_data
                        // reads getattr, not __dict__: native prebuilt views
                        // and data descriptors must be observed before defaults.
                        use crate::flowspace::model::{HostGetAttrError, host_getattr};
                        let probe = match _value.map(|host| host_getattr(host, name)) {
                            Some(Ok(value)) => Some(value),
                            None | Some(Err(HostGetAttrError::Missing)) => None,
                            Some(Err(HostGetAttrError::Unsupported)) => {
                                return Err(TyperError::message(format!(
                                    "InstanceRepr.initialize_prebuilt_data: unsupported getattr({name:?})"
                                )));
                            }
                        };
                        if let Some(attrvalue) = probe {
                            let const_for_field =
                                (r.as_ref() as &dyn Repr).convert_const(&attrvalue)?;
                            constant_to_lowlevel_value(&const_for_field)?
                        } else {
                            // upstream rclass.py:959-968:
                            //     attrvalue = self.classdef.classdesc.read_attribute(
                            //         name, None)
                            //     if attrvalue is None:
                            //         llattrvalue = r.lowleveltype._defl()
                            //     else:
                            //         llattrvalue = r.convert_desc_or_const(attrvalue)
                            //
                            // `ClassDictEntry::Constant(c)` ↔ upstream
                            // `Constant(value)` (classdesc.py), and
                            // `ClassDictEntry::Desc(d)` ↔ upstream
                            // `funcdesc` stored verbatim (classdesc.py).
                            // Both arms project cleanly into `DescOrConst`,
                            // matching upstream `convert_desc_or_const`
                            // (rmodel.py).
                            match super::super::super::annotator::classdesc::ClassDesc::read_attribute(
                                &classdesc, name,
                            ) {
                                Some(class_entry) => {
                                    let desc_or_const = match class_entry {
                                        crate::annotator::classdesc::ClassDictEntry::Constant(c) => {
                                            super::rmodel::DescOrConst::Const(c)
                                        }
                                        crate::annotator::classdesc::ClassDictEntry::Desc(d) => {
                                            super::rmodel::DescOrConst::Desc(d)
                                        }
                                    };
                                    let const_for_field =
                                        (r.as_ref() as &dyn Repr).convert_desc_or_const(&desc_or_const)?;
                                    constant_to_lowlevel_value(&const_for_field)?
                                }
                                None => r.lowleveltype()._defl(),
                            }
                        }
                    };
                    if path.is_empty() {
                        result
                            .setattr(mangled_name, llattrvalue)
                            .map_err(TyperError::message)?;
                    } else {
                        result
                            .setattr_at_path(path, mangled_name, llattrvalue)
                            .map_err(TyperError::message)?;
                    }
                }
            }
        } else {
            // upstream rclass.py:973-975 — root sets typeptr.
            // `result.typeptr = rclass.getvtable()`.
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.initialize_prebuilt_data: rtyper weak ref expired",
                )
            })?;
            let rclass = getclassrepr_arc(&rtyper, classdef)?;
            // The vtable is already cast to typeptr (CLASSTYPE) by
            // `cast_vtable_to_typeptr` inside `getvtable`.
            let vtable_ptr = match &rclass {
                ClassReprArc::Inst(inst) => inst.getvtable()?,
                ClassReprArc::Root(root) => root.getvtable()?,
            };
            let val = lltype::LowLevelValue::Ptr(Box::new(vtable_ptr));
            if path.is_empty() {
                result
                    .setattr("typeptr", val)
                    .map_err(TyperError::message)?;
            } else {
                result
                    .setattr_at_path(path, "typeptr", val)
                    .map_err(TyperError::message)?;
            }
        }
        Ok(())
    }

    /// RPython `InstanceRepr.convert_const_exact(self, value)`
    /// (rclass.py):
    ///
    /// ```python
    /// def convert_const_exact(self, value):
    ///     try:
    ///         return self.iprebuiltinstances[value]
    ///     except KeyError:
    ///         self.setup()
    ///         result = self.create_instance()
    ///         self.iprebuiltinstances[value] = result
    ///         self.initialize_prebuilt_instance(value, self.classdef, result)
    ///         return result
    /// ```
    ///
    /// `iprebuiltinstances` is the value-keyed identity cache
    /// (rclass.py); upstream uses `identity_dict()` which is
    /// pointer-keyed dict semantics — pyre keys on
    /// [`HostObject`]'s Arc identity (Hash + Eq via `Arc::ptr_eq`).
    /// `initialize_prebuilt_instance` queues nested initialization through
    /// the shared FlattenRecursion, as upstream does for circular prebuilt
    /// graphs and their hash initialization order.
    pub fn convert_const_exact(
        self: &Arc<Self>,
        host_obj: &HostObject,
    ) -> Result<Constant, TyperError> {
        // upstream `try: return self.iprebuiltinstances[value]`.
        if let Some(cached) = self.iprebuiltinstances.borrow().get(host_obj).cloned() {
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(cached)),
                self.lowleveltype.clone(),
            ));
        }
        // upstream `self.setup()`. Repr::setup is idempotent.
        Repr::setup(self.as_ref() as &dyn Repr)?;
        // upstream rclass.py:799-800:
        //     result = self.create_instance()
        //     self.iprebuiltinstances[value] = result
        //     self.initialize_prebuilt_data(value, self.classdef, result)
        //
        // RPython aliases `result` between the cache slot and the
        // init target via Python object identity, so mutations made
        // by `initialize_prebuilt_data` are visible through
        // `iprebuiltinstances[value]` for recursive
        // `convert_const(value)` calls during init (`a.b.back == a`
        // intra-class circular prebuilt instances).
        //
        // Pyre clones `result` into the cache; both the cached clone
        // and the init target (`local`, the moved original) share
        // `_struct._fields` / `_array.items` via `Arc<Mutex<...>>`
        // (lltype.rs), preserving the same
        // mid-init aliasing semantics. The cache `borrow_mut` is
        // dropped before init runs so a recursive `convert_const_exact`
        // cache probe at the top of this function can re-acquire
        // `borrow()` without panicking on overlapping borrows.
        let initial = self.create_instance()?;
        self.iprebuiltinstances
            .borrow_mut()
            .insert(host_obj.clone(), initial.clone());
        let local = initial;
        self.initialize_prebuilt_instance(host_obj, self.classdef.as_ref(), &local)?;
        Ok(Constant::with_concretetype(
            ConstValue::LLPtr(Box::new(local)),
            self.lowleveltype.clone(),
        ))
    }

    /// RPython `InstanceRepr.initialize_prebuilt_instance`: defer nested
    /// initialization until the outer object's fields (including hash state)
    /// have been populated. Pointer clones share the allocated field storage.
    pub fn initialize_prebuilt_instance(
        self: &Arc<Self>,
        value: &HostObject,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        result: &_ptr,
    ) -> Result<(), TyperError> {
        let repr = self.clone();
        let value = value.clone();
        let classdef = classdef.cloned();
        let mut result = result.clone();
        INITIALIZE_DATA_FLATTENREC.with(|flat| {
            flat.call(Box::new(move || {
                repr.initialize_prebuilt_data(Some(&value), classdef.as_ref(), &mut result, &[])
            }))
        })
    }

    /// RPython `InstanceRepr.get_reusable_prebuilt_instance(self)`
    /// (rclass.py):
    ///
    /// ```python
    /// def get_reusable_prebuilt_instance(self):
    ///     try:
    ///         return self._reusable_prebuilt_instance
    ///     except AttributeError:
    ///         self.setup()
    ///         result = self.create_instance()
    ///         self._reusable_prebuilt_instance = result
    ///         self.initialize_prebuilt_data(Ellipsis, self.classdef, result)
    ///         return result
    /// ```
    ///
    /// Lazy-cached `_ptr` to a single dummy instance, used by
    /// `ExceptionData.get_standard_ll_exc_instance` to materialise the
    /// canonical instance for a given exception class. Multiple calls
    /// return the same `_ptr` value (same `_identity`).
    pub fn get_reusable_prebuilt_instance(self: &Arc<Self>) -> Result<_ptr, TyperError> {
        if let Some(cached) = self.reusable_prebuilt_instance.borrow().clone() {
            return Ok(cached);
        }
        // upstream: `self.setup()`. Repr::setup is idempotent.
        Repr::setup(self.as_ref() as &dyn Repr)?;
        // upstream rclass.py:810-813:
        //     result = self.create_instance()
        //     self._reusable_prebuilt_instance = result
        //     self.initialize_prebuilt_data(Ellipsis, self.classdef, result)
        //     return result
        //
        // Pyre clones `result` into the cache; the cached clone and the
        // init target (`local`, the moved original) share
        // `_struct._fields` / `_array.items` via `Arc<Mutex<...>>`
        // (lltype.rs), so mutations applied by
        // `initialize_prebuilt_data` are visible through
        // `_reusable_prebuilt_instance` for the no-recursion case
        // (`get_standard_ll_exc_instance`). The cache `borrow_mut` is
        // dropped before init runs.
        let result = self.create_instance()?;
        *self.reusable_prebuilt_instance.borrow_mut() = Some(result.clone());
        let mut local = result;
        self.initialize_prebuilt_data(None, self.classdef.as_ref(), &mut local, &[])?;
        Ok(local)
    }

    /// RPython `InstanceRepr._get_field(self, attr)` — `self.fields[attr]`.
    fn _get_field(&self, attr: &str) -> Option<(String, Arc<dyn Repr>)> {
        self.fields.borrow().get(attr).cloned()
    }

    /// RPython `InstanceRepr._parse_field_list`.
    fn _parse_field_list(
        &self,
        fields: &[String],
        accessor: &mut FieldListAccessor,
        hints: &HashMap<String, ConstValue>,
    ) -> Result<HashMap<String, ImmutableRanking>, TyperError> {
        let mut ranking = HashMap::new();
        for fullname in fields {
            let (name, rank, quasi) = if let Some(name) = fullname.strip_suffix("?[*]") {
                (name, IR_QUASIIMMUTABLE_ARRAY, true)
            } else if let Some(name) = fullname.strip_suffix("[*]") {
                (name, IR_IMMUTABLE_ARRAY, false)
            } else if let Some(name) = fullname.strip_suffix('?') {
                (name, IR_QUASIIMMUTABLE, true)
            } else {
                (fullname.as_str(), IR_IMMUTABLE, false)
            };
            let Some((mangled_name, r)) = self._get_field(name) else {
                continue;
            };
            // `rclass.py _parse_field_list`: `isinstance(r, AbstractBaseListRepr)`.
            // Both `ListRepr` and `FixedSizeListRepr` inherit that base
            // (`rlist.py AbstractFixedSizeListRepr(AbstractBaseListRepr)`);
            // `[*]` / `?[*]` therefore accepts a never-resized list
            // (`Ptr(GcArray)`), not only the resized `ListRepr` header.
            if matches!(rank, IR_IMMUTABLE_ARRAY | IR_QUASIIMMUTABLE_ARRAY)
                && !matches!(
                    r.repr_class_id(),
                    ReprClassId::ListRepr | ReprClassId::FixedSizeListRepr
                )
            {
                return Err(TyperError::message(format!(
                    "_immutable_fields_ = [{fullname:?}] in {:?}, but {name:?} is not a list \
                     (got {})",
                    self.classdef.as_ref().map(|cd| cd.borrow().name.clone()),
                    r.class_name(),
                )));
            }
            if quasi && hints.get("immutable") == Some(&ConstValue::Bool(true)) {
                return Err(TyperError::message(format!(
                    "can't have _immutable_ = True and a quasi-immutable field {name} \
                     in class {:?}",
                    self.classdef.as_ref().map(|cd| cd.borrow().name.clone())
                )));
            }
            ranking.insert(mangled_name, rank);
        }
        // `FieldListAccessor.initialize(self.object_type, ranking)` after
        // `object_type.become(Struct)`. A still-forward `object_type` is
        // the pre-`become` hint walk; `resolved()` is the Struct
        // `test_rvirtualizable.TestVirtualizable.test_accessor` compares.
        let type_for_accessor = match &self.object_type {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().unwrap_or_else(|| self.object_type.clone())
            }
            other => other.clone(),
        };
        accessor.initialize(type_for_accessor, ranking.clone());
        Ok(ranking)
    }

    fn finish_virtualizable_setup(&self) -> Result<(), TyperError> {
        let Some(mut vable) = self.virtualizable.borrow().clone() else {
            return Ok(());
        };
        if vable.top_of_virtualizable_hierarchy {
            let classdesc = self
                .classdef
                .as_ref()
                .map(|cd| cd.borrow().classdesc.clone())
                .ok_or_else(|| {
                    TyperError::message(
                        "VirtualizableInstanceRepr._setup_repr needs a classdesc".to_string(),
                    )
                })?;
            let vfields = classdesc.borrow().get_param("_virtualizable_", None, false);
            let names = const_value_field_names(&vfields);
            // `_parse_field_list` only consults `hints['immutable']`.
            // The accessor object itself is stamped on the struct type
            // as `virtualizable_accessor` — see `virtualizable_accessor_hint`.
            let ranking =
                self._parse_field_list(&names, &mut vable.accessor.lock(), &HashMap::new())?;
            vable.my_redirected_fields = ranking.into_keys().map(|name| (name, true)).collect();
        } else if let Some(base) = self.rbase.borrow().as_ref()
            && let Some(base_vable) = base.virtualizable.borrow().clone()
        {
            vable.my_redirected_fields = base_vable.my_redirected_fields;
        }
        *self.virtualizable.borrow_mut() = Some(vable);
        Ok(())
    }

    /// First step of `VirtualizableInstanceRepr._setup_repr`:
    /// `hints = {'virtualizable_accessor': self.accessor}` passed to
    /// `MkStruct`. The live `FieldListAccessor` is the hint value;
    /// `_parse_field_list` later initializes that same object.
    fn virtualizable_accessor_hint(&self) -> Result<Vec<(String, ConstValue)>, TyperError> {
        let Some(vable) = self.virtualizable.borrow().clone() else {
            return Ok(vec![]);
        };
        if !vable.top_of_virtualizable_hierarchy {
            return Ok(vec![]);
        }
        Ok(vec![(
            "virtualizable_accessor".to_string(),
            ConstValue::Opaque(crate::flowspace::model::OpaqueConst::from_arc(
                std::sync::Arc::clone(&vable.accessor),
            )),
        )])
    }
}

/// Build the `(object_type, lowleveltype)` pair for an InstanceRepr
/// per rclass.py.
fn build_instance_types(
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    gcflavor: Flavor,
) -> (LowLevelType, LowLevelType) {
    if classdef.is_some() {
        // rclass.py:474-475 — `ForwardRef = lltype.FORWARDREF_BY_FLAVOR[
        // LLFLAVOR[gcflavor]]; self.object_type = ForwardRef()`.
        let fwd = match gcflavor {
            Flavor::Gc => ForwardReference::gc(),
            Flavor::Raw => ForwardReference::new(),
        };
        let fwd_for_ptr = fwd.clone();
        let object_type = LowLevelType::ForwardReference(Box::new(fwd));
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd_for_ptr),
        }));
        (object_type, lowleveltype)
    } else {
        // rclass.py:471-472 — classdef is None path.
        let object_type = object_by_flavor(gcflavor);
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: match &object_type {
                LowLevelType::Struct(body) => PtrTarget::Struct((**body).clone()),
                other => panic!("object_by_flavor returned non-Struct {other:?}"),
            },
        }));
        (object_type, lowleveltype)
    }
}

impl Repr for InstanceRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "InstanceRepr"
    }

    /// RPython `InstanceRepr.gcflavor` (rclass.py) — surface the
    /// per-class flavor as the upstream-string form ('gc' / 'raw') so
    /// callers (`rbuiltin.rtype_free_non_gc_object`) can read it
    /// without an `std::any::Any` downcast.
    fn gc_flavor_str(&self) -> Option<&'static str> {
        Some(self.gcflavor.llflavor())
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::InstanceRepr
    }

    /// RPython `InstanceRepr.rtype_type(self, hop)` (rclass.py):
    ///
    /// ```python
    /// def rtype_type(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     instance_repr = self.common_repr()
    ///     vinst, = hop.inputargs(instance_repr)
    ///     if hop.args_s[0].can_be_none():
    ///         return hop.gendirectcall(ll_inst_type, vinst)
    ///     else:
    ///         return instance_repr.getfield(vinst, '__class__', hop.llops)
    /// ```
    fn rtype_type(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeObjectTrait;
        use crate::translator::rtyper::rtyper::ConvertedTo;

        // upstream: `if hop.s_result.is_constant(): ...`.
        let s_result = hop
            .s_result
            .borrow()
            .clone()
            .ok_or_else(|| TyperError::message("InstanceRepr.rtype_type: s_result missing"))?;
        if s_result.is_constant() {
            let r_result =
                hop.r_result.borrow().clone().ok_or_else(|| {
                    TyperError::message("InstanceRepr.rtype_type: r_result missing")
                })?;
            let value = s_result
                .const_()
                .expect("s_result.is_constant() implies const_() is Some");
            return HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), value)
                .map(Hlvalue::Constant)
                .map(Some);
        }

        // upstream: `instance_repr = self.common_repr();
        //            vinst, = hop.inputargs(instance_repr)`.
        let instance_repr = self.common_repr()?;
        let mut vinst =
            hop.inputargs(vec![ConvertedTo::Repr(instance_repr.as_ref() as &dyn Repr)])?;
        let vinst = vinst
            .pop()
            .expect("one converted argument for unary type operation");

        // upstream: `if hop.args_s[0].can_be_none():
        //                return hop.gendirectcall(ll_inst_type, vinst)`.
        let can_be_none = hop
            .args_s
            .borrow()
            .first()
            .ok_or_else(|| TyperError::message("InstanceRepr.rtype_type: args_s[0] missing"))?
            .can_be_none();
        if can_be_none {
            let rtyper = self.rtyper.upgrade().ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_type: rtyper weak ref expired")
            })?;
            let helper = rtyper.lowlevel_helper_function(
                "ll_inst_type",
                vec![instance_repr.lowleveltype().clone()],
                CLASSTYPE.clone(),
            )?;
            return hop.gendirectcall(&helper, vec![vinst]);
        }

        // upstream non-null arm: read the root instance's `__class__`
        // field directly.
        let mut llops = hop.llops.borrow_mut();
        let vtype =
            instance_repr.getfield(vinst, "__class__", &mut llops, false, &Flags::default())?;
        Ok(Some(Hlvalue::Variable(vtype)))
    }

    /// RPython `InstanceRepr.rtype_getattr(self, hop)` (rclass.py):
    ///
    /// ```python
    /// def rtype_getattr(self, hop):
    ///     if hop.s_result.is_constant():
    ///         return hop.inputconst(hop.r_result, hop.s_result.const)
    ///     attr = hop.args_s[1].const
    ///     vinst, vattr = hop.inputargs(self, Void)
    ///     if attr == '__class__' and hop.r_result.lowleveltype is Void:
    ///         # special case for when the result of '.__class__' is a constant
    ///         [desc] = hop.s_result.descriptions
    ///         return hop.inputconst(Void, desc.pyobj)
    ///     if attr in self.allinstancefields:
    ///         return self.getfield(vinst, attr, hop.llops,
    ///                              flags=hop.args_s[0].flags)
    ///     elif attr in self.rclass.allmethods:
    ///         # special case for methods: represented as their 'self' only
    ///         # (see MethodsPBCRepr)
    ///         return hop.r_result.get_method_from_instance(self, vinst,
    ///                                                      hop.llops)
    ///     else:
    ///         vcls = self.getfield(vinst, '__class__', hop.llops)
    ///         return self.rclass.getclsfield(vcls, attr, hop.llops)
    /// ```
    ///
    /// Step 6 (`allmethods` method dispatch) routes through
    /// `MethodsPBCRepr.get_method_from_instance` (rpbc.py)
    /// which is also unported.  For `classdef=None` (the only
    /// `InstanceRepr` flavour reaching the rtyper today, per
    /// `valuetype_to_someshell(Ref) -> SomeInstance(classdef=None)` at
    /// `flowspace_adapter.rs`), the Step 6 branch is
    /// unreachable: `RootClassRepr::_setup_repr` keeps
    /// `allmethods = {}`.
    fn rtype_getattr(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::{SomeObjectTrait, SomeValue};
        use crate::flowspace::model::Hlvalue;
        use crate::translator::rtyper::rtyper::ConvertedTo;

        // upstream rclass.py:839-840: `if hop.s_result.is_constant():
        //     return hop.inputconst(hop.r_result, hop.s_result.const)`.
        let s_result_clone = hop.s_result.borrow().clone();
        if let Some(s_result) = &s_result_clone
            && s_result.is_constant()
        {
            let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: r_result missing")
            })?;
            let const_val = s_result
                .const_()
                .cloned()
                .expect("s_result.is_constant() implies const_() is Some");
            return HighLevelOp::inputconst(ConvertedTo::Repr(r_result.as_ref()), &const_val)
                .map(Hlvalue::Constant)
                .map(Some);
        }

        // upstream: `attr = hop.args_s[1].const` (rclass.py:841).
        let attr = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: hop.args_s[1] missing")
            })?;
            s_attr
                .const_()
                .and_then(ConstValue::as_pystr)
                .ok_or_else(|| {
                    TyperError::message(
                        "InstanceRepr.rtype_getattr: attribute name must be a constant string",
                    )
                })?
                .to_string()
        };

        // upstream: `vinst, vattr = hop.inputargs(self, Void)` (rclass.py).
        let void = LowLevelType::Void;
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self), ConvertedTo::from(&void)])?;
        let vinst = vlist[0].clone();

        // Ptr bound method on a pointer-carrying instance: the
        // annotator answers `getattr(p, "is_null")` with a
        // `SomeBuiltinMethod` the way `SomePtr.getattr` would
        // (unaryop.rs `ptr_method_is_null`; receivers that upstream
        // keeps as `SomePtr` are carried as `SomeInstance` by pyre's
        // pointer erasure). Mirror `PtrRepr.rtype_getattr`'s
        // ADT-method pass-through (rptr.py `if isinstance(
        // hop.s_result, SomeLLADTMeth): return hop.inputarg(
        // hop.r_result, arg=0)`): the bound method is represented as
        // its receiver (`BuiltinMethodRepr.lowleveltype ==
        // self_repr.lowleveltype`, rbuiltin.py), so getattr
        // forwards the receiver value unchanged and
        // `BuiltinMethodRepr.rtype_simple_call` dispatches the call
        // back through `rtype_method` below.
        if matches!(s_result_clone.as_ref(), Some(SomeValue::BuiltinMethod(_))) {
            return Ok(Some(vinst));
        }

        // upstream rclass.py:843-846:
        //     if attr == '__class__' and hop.r_result.lowleveltype is Void:
        //         # special case for when the result of '.__class__' is a constant
        //         [desc] = hop.s_result.descriptions
        //         return hop.inputconst(Void, desc.pyobj)
        let r_result_for_classattr =
            hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: r_result missing")
            })?;
        if attr == "__class__"
            && matches!(r_result_for_classattr.lowleveltype(), &LowLevelType::Void)
        {
            let pbc = match s_result_clone.as_ref() {
                Some(SomeValue::PBC(pbc)) => pbc.clone(),
                _ => {
                    return Err(TyperError::message(
                        "InstanceRepr.rtype_getattr: __class__ Void-typed result expects \
                         SomePBC s_result (rclass.py:843-846)",
                    ));
                }
            };
            if pbc.descriptions.len() != 1 {
                return Err(TyperError::message(
                    "InstanceRepr.rtype_getattr: __class__ Void-typed result expects a \
                     singleton SomePBC.descriptions (rclass.py:845)",
                ));
            }
            let entry = pbc.descriptions.values().next().unwrap();
            let pyobj = entry.pyobj().ok_or_else(|| {
                TyperError::message(
                    "InstanceRepr.rtype_getattr: __class__ description missing pyobj \
                     (rclass.py:846)",
                )
            })?;
            return HighLevelOp::inputconst(
                ConvertedTo::from(&LowLevelType::Void),
                &ConstValue::HostObject(pyobj),
            )
            .map(Hlvalue::Constant)
            .map(Some);
        }

        // Recover `Arc<Self>` from the rtyper repr cache so we can call
        // the `&Arc<Self>`-receiver helpers (`getfield`).  Same pattern
        // as `InstanceRepr::convert_const` which routes back through
        // `getinstancerepr`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.rtype_getattr: rtyper weak ref expired")
        })?;
        let self_arc = getinstancerepr(&rtyper, self.classdef.as_ref(), self.gcflavor)?;

        // upstream: `if attr in self.allinstancefields:`
        //          `return self.getfield(vinst, attr, hop.llops,
        //                                flags=hop.args_s[0].flags)`
        // (rclass.py).
        let has_field = self.allinstancefields.borrow().contains_key(&attr);
        if has_field {
            let flags = {
                let args_s = hop.args_s.borrow();
                let s_inst = args_s.first().ok_or_else(|| {
                    TyperError::message("InstanceRepr.rtype_getattr: hop.args_s[0] missing")
                })?;
                match s_inst {
                    SomeValue::Instance(inst) => inst
                        .flags
                        .iter()
                        .map(|(k, v)| (k.clone(), ConstValue::Bool(*v)))
                        .collect::<Flags>(),
                    _ => Flags::default(),
                }
            };
            let mut llops = hop.llops.borrow_mut();
            let var = self_arc.getfield(vinst, &attr, &mut llops, false, &flags)?;
            return Ok(Some(Hlvalue::Variable(var)));
        }

        // upstream: `elif attr in self.rclass.allmethods:`
        //          `return hop.r_result.get_method_from_instance(self, vinst, hop.llops)`
        // (rclass.py:850-854).
        let rclass = self.rclass.borrow().clone().ok_or_else(|| {
            TyperError::message("InstanceRepr.rtype_getattr: rclass missing — call setup() first")
        })?;
        let has_method = match &rclass {
            ClassReprArc::Inst(c) => c.allmethods().contains_key(&attr),
            // `RootClassRepr.allmethods` is always empty (`RootClassRepr::_setup_repr`).
            ClassReprArc::Root(_) => false,
        };
        if has_method {
            // upstream rclass.py:850-854:
            //     elif attr in self.rclass.allmethods:
            //         # special case for methods: represented as their
            //         # 'self' only (see MethodsPBCRepr)
            //         return hop.r_result.get_method_from_instance(
            //             self, vinst, hop.llops)
            //
            // `hop.r_result` upstream is the `MethodsPBCRepr` for the
            // dispatched method.  Pyre stores it as `Arc<dyn Repr>`;
            // downcast through `Repr: Any` (rmodel.rs) to reach the
            // concrete `get_method_from_instance` (rpbc.rs).
            use crate::translator::rtyper::rpbc::MethodsPBCRepr;
            let r_result_arc = hop.r_result.borrow().clone().ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_getattr: r_result missing on method branch")
            })?;
            let methods_repr = (r_result_arc.as_ref() as &dyn std::any::Any)
                .downcast_ref::<MethodsPBCRepr>()
                .ok_or_else(|| {
                    TyperError::message(format!(
                        "InstanceRepr.rtype_getattr: method '{attr}' dispatch expects \
                         MethodsPBCRepr r_result, got {}",
                        r_result_arc.class_name()
                    ))
                })?;
            let mut llops = hop.llops.borrow_mut();
            let v_method = methods_repr.get_method_from_instance(
                self_arc.as_ref() as &dyn Repr,
                vinst.clone(),
                &mut llops,
            )?;
            return Ok(Some(v_method));
        }

        // A niche `Option<&T>` receiver collapses to a maybe-null instance
        // pointer with no `__discriminant` tag field — a payload enum's
        // `__discriminant` IS an instance field, resolved by the
        // `allinstancefields` branch above.  Reaching the fall-through for
        // `"__discriminant"` therefore means the discriminant is the
        // pointer's null-ness (`None` = null = 0, `Some` = non-null = 1), so
        // read it as `ptr_nonzero(vinst)` cast to the result lltype:
        // `cast_bool_to_uint` when that lltype is `Unsigned` or
        // `UnsignedLongLong`, and `cast_bool_to_int` otherwise, rather than a
        // class field (which does not exist and rtypes to Void).  Mirrors the
        // `CanBeNull.rtype_bool` `ptr_nonzero` lowering (rmodel.py) and the
        // front-end `Discriminant` niche fold's in-place null test; the
        // `option_is_none` post-pass emits this `__discriminant` FieldRead for
        // both niche and aggregate `is_some`/`is_none` receivers.
        //
        // Gate on the receiver being a genuinely nullable instance
        // (`SomeInstance.can_be_none`): only a niche `Option<&T>` reads its
        // discriminant from the pointer's null-ness.  A statically non-null
        // receiver — or one whose tag field should exist but is absent through
        // an unpopulated `allinstancefields`, a mangled-name mismatch, or a
        // >2-variant enum erased to a bare pointer — must fall through to the
        // `getclsfield` path below and surface a `TyperError`, matching
        // upstream (rclass.py raises on a missing field), rather than
        // fabricating a 0/1 value.
        let receiver_is_nullable_instance = {
            let args_s = hop.args_s.borrow();
            matches!(
                args_s.first(),
                Some(SomeValue::Instance(inst)) if inst.can_be_none
            )
        };
        if attr == "__discriminant" && receiver_is_nullable_instance {
            use crate::translator::rtyper::rtyper::GenopResult;
            // Result lltype follows `hop.r_result`: a switch on tags
            // 0..n specialises `SomeInteger` to Ruint/Unsigned; a
            // discriminant that never gained knowntypedata stays Signed.
            let disc_lltype = hop
                .r_result
                .borrow()
                .as_ref()
                .map(|r| r.lowleveltype().clone())
                .unwrap_or(LowLevelType::Signed);
            let cast_op = match &disc_lltype {
                LowLevelType::Unsigned | LowLevelType::UnsignedLongLong => "cast_bool_to_uint",
                _ => "cast_bool_to_int",
            };
            let v_nonzero = hop
                .genop(
                    "ptr_nonzero",
                    vec![vinst.clone()],
                    GenopResult::LLType(LowLevelType::Bool),
                )
                .ok_or_else(|| {
                    TyperError::message(
                        "InstanceRepr.rtype_getattr: ptr_nonzero produced no result var",
                    )
                })?;
            return Ok(hop.genop(cast_op, vec![v_nonzero], GenopResult::LLType(disc_lltype)));
        }

        // upstream: `else:`
        //          `vcls = self.getfield(vinst, '__class__', hop.llops)`
        //          `return self.rclass.getclsfield(vcls, attr, hop.llops)`
        // (rclass.py).
        let mut llops = hop.llops.borrow_mut();
        let flags = Flags::default();
        let v_cls = self_arc.getfield(vinst, "__class__", &mut llops, false, &flags)?;
        let var = match &rclass {
            ClassReprArc::Inst(c) => c.getclsfield(Hlvalue::Variable(v_cls), &attr, &mut llops)?,
            ClassReprArc::Root(r) => r.getclsfield(Hlvalue::Variable(v_cls), &attr, &mut llops)?,
        };
        Ok(Some(Hlvalue::Variable(var)))
    }

    /// RPython `InstanceRepr.rtype_setattr(self, hop)` (rclass.py):
    ///
    /// ```python
    /// def rtype_setattr(self, hop):
    ///     attr = hop.args_s[1].const
    ///     r_value = self.getfieldrepr(attr)
    ///     vinst, vattr, vvalue = hop.inputargs(self, Void, r_value)
    ///     self.setfield(vinst, attr, vvalue, hop.llops,
    ///                   flags=hop.args_s[0].flags)
    /// ```
    fn rtype_setattr(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::annotator::model::SomeValue;
        use crate::translator::rtyper::rtyper::ConvertedTo;

        // upstream: `attr = hop.args_s[1].const`.
        let (attr, flags) = {
            let args_s = hop.args_s.borrow();
            let s_attr = args_s.get(1).ok_or_else(|| {
                TyperError::message("InstanceRepr.rtype_setattr: hop.args_s[1] missing")
            })?;
            let attr = s_attr
                .const_()
                .and_then(ConstValue::as_pystr)
                .ok_or_else(|| {
                    TyperError::message(
                        "InstanceRepr.rtype_setattr: attribute name must be a constant string",
                    )
                })?
                .to_string();
            // upstream: `flags=hop.args_s[0].flags`.
            let flags = match args_s.first() {
                Some(SomeValue::Instance(inst)) => inst
                    .flags
                    .iter()
                    .map(|(k, v)| (k.clone(), ConstValue::Bool(*v)))
                    .collect::<Flags>(),
                _ => Flags::default(),
            };
            (attr, flags)
        };

        // Recover `Arc<Self>` for the `&Arc<Self>`-receiver helpers
        // (`getfieldrepr` / `setfield`), same as `rtype_getattr`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.rtype_setattr: rtyper weak ref expired")
        })?;
        let self_arc = getinstancerepr(&rtyper, self.classdef.as_ref(), self.gcflavor)?;

        // upstream: `r_value = self.getfieldrepr(attr)`.
        let r_value = self_arc.getfieldrepr(&attr)?;
        // upstream: `vinst, vattr, vvalue = hop.inputargs(self, Void, r_value)`.
        let void = LowLevelType::Void;
        let vlist = hop.inputargs(vec![
            ConvertedTo::Repr(self),
            ConvertedTo::from(&void),
            ConvertedTo::Repr(r_value.as_ref()),
        ])?;
        // upstream: `self.setfield(vinst, attr, vvalue, hop.llops, flags=...)`.
        let mut llops = hop.llops.borrow_mut();
        self_arc.setfield(
            vlist[0].clone(),
            &attr,
            vlist[2].clone(),
            &mut llops,
            false,
            &flags,
        )?;
        Ok(None)
    }

    /// RPython `InstanceRepr.rtype_bool(self, hop)` (rclass.py):
    ///
    /// ```python
    /// def rtype_bool(self, hop):
    ///     vinst, = hop.inputargs(self)
    ///     return hop.genop('ptr_nonzero', [vinst], resulttype=Bool)
    /// ```
    ///
    /// `InstanceRepr` does not inherit from `CanBeNull` upstream, so the
    /// constant fast-path in [`crate::translator::rtyper::rmodel::can_be_null_rtype_bool`]
    /// is intentionally absent here — a non-null instance always evaluates
    /// to true and the lowleveltype carrier is a `Ptr(GcStruct)` that
    /// `ptr_nonzero` lowers directly to a null comparison.
    /// RPython `InstanceRepr.get_ll_eq_function` (`rclass.py`): `None`,
    /// so a dict key compares by pointer identity.
    fn get_ll_eq_function(
        &self,
        _rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `InstanceRepr.get_ll_hash_function` (`rclass.py`):
    /// `ll_inst_hash`.
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let arg = self.lowleveltype.clone();
        rtyper
            .lowlevel_helper_function_with_builder(
                "ll_inst_hash",
                vec![arg],
                LowLevelType::Signed,
                |_rtyper, args, _result| build_ll_inst_hash_helper_graph("ll_inst_hash", &args[0]),
            )
            .map(Some)
    }

    /// RPython `InstanceRepr.get_ll_fasthash_function = get_ll_hash_function`
    /// (`rclass.py`).
    fn get_ll_fasthash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        self.get_ll_hash_function(rtyper)
    }

    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult};
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
        Ok(hop.genop(
            "ptr_nonzero",
            vlist,
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }

    /// `BuiltinMethodRepr.rtype_simple_call` dispatch target for the
    /// ptr bound method the annotator seats on pointer-carrying
    /// instances (`unaryop.rs ptr_method_is_null`; see the
    /// `SomeBuiltinMethod` pass-through in [`Self::rtype_getattr`]).
    /// `is_null` is the lltype `_ptr` nullity probe — lower it as
    /// `ptr_iszero` (opimpl.py `op_ptr_iszero`), the negated
    /// twin of `rtype_bool`'s `ptr_nonzero` above.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult};
        if method_name == "is_null" {
            let vlist = hop.inputargs(vec![ConvertedTo::Repr(self)])?;
            return Ok(hop.genop("ptr_iszero", vlist, GenopResult::LLType(LowLevelType::Bool)));
        }
        // `Option<RawStruct>::as_ref` / `<*mut T>::as_ref` on a
        // pointer-carrying instance: the receiver is already the
        // nullable pointer word (`ptr_method_as_ref`). Identity.
        if method_name == "as_ref" || method_name == "as_mut" {
            return hop.inputarg(self, 0).map(Some);
        }
        if method_name == "to_exc_object" {
            let is_pyerror = self.classdef.as_ref().is_some_and(|classdef| {
                classdef.borrow().name.rsplit('.').next() == Some("PyError")
            });
            if is_pyerror {
                use crate::translator::rtyper::lltypesystem::lltype::{
                    self, FuncType, functionptr_with_external_name,
                };

                // Pyre's `PyError::to_exc_object` materializes the trace-level
                // exception value (`error.rs`), matching the
                // `exceptiontransform.py:347-354` decision point where RPython
                // raises the normalized exception object.  It is an interpreter
                // impl method, not an rtyper-minted helper, so keep it as a
                // residual `direct_call` keyed by
                // `CallPath::for_impl_method("PyError", "to_exc_object")`.
                let r_result = hop.r_result.borrow().clone().ok_or_else(|| {
                    TyperError::message("InstanceRepr.rtype_method_to_exc_object: r_result missing")
                })?;
                let arg_lltype = self.lowleveltype().clone();
                let result_lltype = r_result.lowleveltype().clone();
                let funcptr = functionptr_with_external_name(
                    FuncType {
                        args: vec![arg_lltype],
                        result: result_lltype,
                    },
                    "PyError::to_exc_object",
                    Some("PyError.to_exc_object".to_string()),
                );
                let funcptr_lltype = LowLevelType::Ptr(Box::new(lltype::typeOf(&funcptr)));
                let v_func = HighLevelOp::inputconst(
                    &funcptr_lltype,
                    &ConstValue::LLPtr(Box::new(funcptr)),
                )?;
                let mut vlist = vec![Hlvalue::Constant(v_func)];
                vlist.extend(hop.inputargs(vec![ConvertedTo::Repr(self)])?);
                hop.exception_cannot_occur()?;
                return Ok(hop.genop("direct_call", vlist, GenopResult::Repr(r_result)));
            }
        }
        Err(TyperError::message(format!(
            "missing {}.rtype_method_{method_name}",
            self.class_name()
        )))
    }

    /// RPython `InstanceRepr.rtype_isinstance(self, hop)` (rclass.py):
    ///
    /// ```python
    /// def rtype_isinstance(self, hop):
    ///     class_repr = get_type_repr(hop.rtyper)
    ///     instance_repr = self.common_repr()
    ///
    ///     v_obj, v_cls = hop.inputargs(instance_repr, class_repr)
    ///     if isinstance(v_cls, Constant):
    ///         cls = v_cls.value
    ///         llf, llf_nonnull = make_ll_isinstance(self.rtyper, cls)
    ///         if hop.args_s[0].can_be_None:
    ///             return hop.gendirectcall(llf, v_obj)
    ///         else:
    ///             return hop.gendirectcall(llf_nonnull, v_obj)
    ///     else:
    ///         return hop.gendirectcall(ll_isinstance, v_obj, v_cls)
    /// ```
    ///
    /// Constant `v_cls` is recognised here and routed through
    /// [`make_ll_isinstance`] (rclass.py) to pick up the
    /// per-class `int_between` / `ptr_eq` specialisation. The choice
    /// between `ll_isinstance_const` (null-checking) and
    /// `ll_isinstance_const_nonnull` (assumes nonnull) is driven by
    /// `hop.args_s[0].can_be_None`. Variable `v_cls` falls through to
    /// the unspecialised `ll_isinstance` helper minted by E.1.
    fn rtype_isinstance(&self, hop: &HighLevelOp) -> RTypeResult {
        use crate::translator::rtyper::rtyper::{ConvertedTo, make_ll_isinstance};

        // upstream: `class_repr = get_type_repr(hop.rtyper)`.
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.rtype_isinstance: rtyper weak ref expired")
        })?;
        let class_repr = get_type_repr(&rtyper)?;
        // upstream: `instance_repr = self.common_repr()`.
        let instance_repr = self.common_repr()?;

        // upstream: `v_obj, v_cls = hop.inputargs(instance_repr, class_repr)`.
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(instance_repr.as_ref() as &dyn Repr),
            ConvertedTo::Repr(class_repr.as_ref()),
        ])?;
        let v_obj = v_args[0].clone();
        let v_cls = v_args[1].clone();

        // upstream: `if isinstance(v_cls, Constant):`.
        if let Hlvalue::Constant(c) = &v_cls {
            // upstream: `cls = v_cls.value`. The constant carries an
            // `_ptr` to the vtable; subclassrange_{min,max} are read
            // off it by make_ll_isinstance.
            let ConstValue::LLPtr(c_ptr) = &c.value else {
                return Err(TyperError::message(format!(
                    "InstanceRepr.rtype_isinstance: constant v_cls must \
                     carry an _ptr, got {:?}",
                    c.value
                )));
            };
            // upstream: `llf, llf_nonnull = make_ll_isinstance(self.rtyper, cls)`.
            // `v_obj` has been coerced through `instance_repr.lowleveltype()`
            // (OBJECTPTR for gc flavor, NONGCOBJECTPTR for raw); pass it
            // through so the minted helper's `obj` parameter type matches
            // the actual `v_obj`'s lltype upstream's "obj should be cast
            // to OBJECT or NONGCOBJECT" polymorphism (rclass.py).
            let (ll_const, ll_const_nonnull) =
                make_ll_isinstance(&rtyper, c_ptr, instance_repr.lowleveltype())?;
            // upstream: `if hop.args_s[0].can_be_None: ...`. The
            // `can_be_None` flag rides on the annotator's
            // SomeInstance / SomePBC carriers — None elsewhere means
            // the obj is statically known non-null, so the nonnull
            // variant skips the runtime branch.
            let can_be_none = {
                let args_s = hop.args_s.borrow();
                match args_s.first() {
                    Some(SomeValue::Instance(inst)) => inst.can_be_none,
                    Some(SomeValue::PBC(pbc)) => pbc.can_be_none,
                    _ => true,
                }
            };
            let helper = if can_be_none {
                ll_const
            } else {
                ll_const_nonnull
            };
            return hop.gendirectcall(&helper, vec![v_obj]);
        }

        // upstream variable case: `gendirectcall(ll_isinstance, v_obj, v_cls)`.
        // `v_obj` was coerced through `instance_repr.lowleveltype()`,
        // which is `OBJECTPTR` for the GC flavor and `NONGCOBJECTPTR`
        // for raw-flavor instances — hard-coding `OBJECTPTR` would
        // mint a helper signature that mismatches the actual argument
        // type for non-GC paths.
        let helper = rtyper.lowlevel_helper_function(
            "ll_isinstance",
            vec![instance_repr.lowleveltype().clone(), CLASSTYPE.clone()],
            LowLevelType::Bool,
        )?;
        hop.gendirectcall(&helper, vec![v_obj, v_cls])
    }

    /// RPython `InstanceRepr.convert_const(self, value)` (rclass.py):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if value is None:
    ///         return self.null_instance()
    ///     if isinstance(value, types.MethodType):
    ///         value = value.im_self   # bound method -> instance
    ///     bk = self.rtyper.annotator.bookkeeper
    ///     try:
    ///         classdef = bk.getuniqueclassdef(value.__class__)
    ///     except KeyError:
    ///         raise TyperError("no classdef: %r" % (value.__class__,))
    ///     if classdef != self.classdef:
    ///         if classdef.commonbase(self.classdef) != self.classdef:
    ///             raise TyperError("not an instance of %r: %r" % (
    ///                 self.classdef.name, value))
    ///         rinstance = getinstancerepr(self.rtyper, classdef)
    ///         result = rinstance.convert_const(value)
    ///         return self.upcast(result)
    ///     # common case
    ///     return self.convert_const_exact(value)
    /// ```
    ///
    /// For the root `object` repr, `self.classdef` is `None`.
    /// RPython `rclass.py` delegates in that arm because
    /// `classdesc.py` makes `classdef.commonbase(None) == None`.
    ///
    /// The `classdef == self.classdef` exact-match arm dispatches into
    /// [`InstanceRepr::convert_const_exact`] (rclass.py) via
    /// [`getinstancerepr`], which recovers the owning `Arc<Self>` from
    /// the rtyper's repr cache so the borrow-cell-backed
    /// `iprebuiltinstances` and `initialize_prebuilt_data` (rclass.py:
    /// 947-975, including the class-level `read_attribute` /
    /// `convert_desc_or_const` branch) can run.
    /// The Variable-arm field-init is in
    /// [`InstanceRepr::initialize_prebuilt_data`].
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        use crate::annotator::classdesc::ClassDef;

        // upstream: `if value is None: return self.null_instance()`.
        if matches!(value, ConstValue::None) {
            let null = self.null_instance()?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(null)),
                self.lowleveltype.clone(),
            ));
        }

        // upstream: `isinstance(value, types.MethodType)` —
        //   bound method redirect to its `im_self`.
        let host_obj = match value {
            ConstValue::HostObject(h) => h.clone(),
            other => {
                let class = self
                    .classdef
                    .as_ref()
                    .map(|cd| cd.borrow().name.clone())
                    .unwrap_or_else(|| "<no classdef>".to_string());
                return Err(TyperError::message(format!(
                    "InstanceRepr.convert_const: expected HostObject or None, \
                     got {other:?} (repr class: {class})"
                )));
            }
        };
        let host_obj = if host_obj.is_bound_method() {
            host_obj.bound_method_self().cloned().ok_or_else(|| {
                TyperError::message("InstanceRepr.convert_const: bound method has no self_obj")
            })?
        } else {
            host_obj
        };

        // upstream: `classdef = bk.getuniqueclassdef(value.__class__)`.
        let host_cls = host_obj.class_of().ok_or_else(|| {
            TyperError::message(format!(
                "InstanceRepr.convert_const: no class_of() for {:?}",
                host_obj.qualname()
            ))
        })?;
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.convert_const: rtyper weak ref dropped")
        })?;
        let annotator = rtyper.annotator.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr.convert_const: annotator weak ref dropped")
        })?;
        let classdef = annotator
            .bookkeeper
            .getuniqueclassdef(&host_cls)
            .map_err(|e| {
                TyperError::message(format!("no classdef: {} ({})", host_cls.qualname(), e))
            })?;

        // upstream: `if classdef != self.classdef`.
        let exact_match = match self.classdef.as_ref() {
            Some(self_cd) => Rc::ptr_eq(self_cd, &classdef),
            None => false,
        };
        if !exact_match {
            // upstream: `if classdef.commonbase(self.classdef) !=
            //              self.classdef: raise TyperError(...)`.
            //   Subclass-relationship check; self must be a base of the
            //   value's class.
            // When `self.classdef` is None — the root `object` InstanceRepr,
            // which pyre's `Ref(None)` erasure lands a known-class constant on
            // — upstream `classdef.commonbase(self.classdef)` returns None (the
            // `while other is not None` loop exits immediately), so `None !=
            // None` is false and the delegate path below applies
            // unconditionally: the root is a base of every class. Only run the
            // subclass-relationship guard for a concrete `self.classdef`.
            if let Some(self_cd) = self.classdef.as_ref() {
                let common = ClassDef::commonbase(&classdef, self_cd).ok_or_else(|| {
                    TyperError::message(format!(
                        "not an instance of {:?}: {:?}",
                        self_cd.borrow().name,
                        host_obj.qualname()
                    ))
                })?;
                if !Rc::ptr_eq(&common, self_cd) {
                    return Err(TyperError::message(format!(
                        "not an instance of {:?}: {:?}",
                        self_cd.borrow().name,
                        host_obj.qualname()
                    )));
                }
            }
            // upstream: `rinstance = getinstancerepr(rtyper, classdef);
            //            result = rinstance.convert_const(value);
            //            return self.upcast(result)`.
            let rinstance = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)?;
            let sub_result = (rinstance.as_ref() as &dyn Repr).convert_const(value)?;
            let ConstValue::LLPtr(sub_ptr) = &sub_result.value else {
                return Err(TyperError::message(format!(
                    "InstanceRepr.convert_const: subclass result is not LLPtr: \
                     {:?}",
                    sub_result.value
                )));
            };
            let upcast_ptr = self.upcast(sub_ptr)?;
            return Ok(Constant::with_concretetype(
                ConstValue::LLPtr(Box::new(upcast_ptr)),
                self.lowleveltype.clone(),
            ));
        }

        // upstream: `return self.convert_const_exact(value)`.
        //
        // The trait method only carries `&self`, but
        // [`convert_const_exact`] needs `&Arc<Self>` for the cache /
        // setup() entry. Look ourselves back up via
        // [`getinstancerepr`] — pyre's repr cache is keyed by
        // `(classdef, flavor)`, which gives the same shared `Arc`
        // back. The lookup is a no-op O(1) HashMap probe (the entry
        // is guaranteed present since `self` exists).
        let arc_self = getinstancerepr(&rtyper, self.classdef.as_ref(), self.gcflavor)?;
        arc_self.convert_const_exact(&host_obj)
    }

    /// RPython `InstanceRepr._setup_repr` (rclass.py).
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr._setup_repr: RPythonTyper weak ref expired")
        })?;

        // rclass.py — `self.rclass = getclassrepr(self.rtyper,
        // self.classdef)`.
        let rclass = getclassrepr_arc(&rtyper, self.classdef.as_ref())?;
        *self.rclass.borrow_mut() = Some(rclass);

        // rclass.py:495-496 — `fields = {}; allinstancefields = {}`.
        // Build local dicts first (rclass.py:488-491 "don't store
        // mutable objects on self before they are fully built").
        let mut fields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let mut allinstancefields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let Some(classdef_rc) = self.classdef.as_ref() else {
            fields.insert(
                "__class__".to_string(),
                ("typeptr".to_string(), get_type_repr(rtyper.as_ref())?),
            );
            allinstancefields.extend(fields.clone());
            *self.fields.borrow_mut() = fields;
            *self.allinstancefields.borrow_mut() = allinstancefields;
            return Ok(());
        };
        let mut myllfields: Vec<(String, LowLevelType)> = Vec::new();

        // rclass.py:501-509 — attrs iteration with readonly filter.
        let attrs_sorted = {
            let cd = classdef_rc.borrow();
            cd.attrs
                .iter()
                .map(|(k, v)| (k.clone(), (v.s_value.clone(), v.readonly)))
                .collect::<BTreeMap<String, (SomeValue, bool)>>()
        };
        for (attr_name, (s_value, readonly)) in &attrs_sorted {
            if *readonly {
                continue;
            }
            let r = rtyper.getrepr(s_value)?;
            let mangled_name = format!("inst_{attr_name}");
            fields.insert(attr_name.clone(), (mangled_name.clone(), r.clone()));
            myllfields.push((mangled_name, r.lowleveltype().clone()));
        }
        sort_llfields(&mut myllfields);

        // rclass.py — `self.rbase = getinstancerepr(
        // self.rtyper, self.classdef.basedef, self.gcflavor);
        // self.rbase.setup()`.
        let basedef = classdef_rc.borrow().basedef.clone();
        let rbase = getinstancerepr(&rtyper, basedef.as_ref(), self.gcflavor)?;
        Repr::setup(rbase.as_ref())?;
        *self.rbase.borrow_mut() = Some(rbase.clone());

        // rclass.py:548-554 — `object_type = MkStruct(classdef.name,
        // ('super', rbase.object_type), hints=hints, adtmeths=adtmeths,
        // *llfields, **kwds)` + `self.object_type.become(object_type)`.
        // Gc-flavor also passes `rtti=True` (rclass.py) so the
        // struct carries an `_runtime_type_info` opaque consumable by
        // `fill_vtable_root` via `getRuntimeTypeInfo`. Immutable /
        // special_memory_pressure hints stay unported (R2-D).
        // `VirtualizableInstanceRepr._setup_repr_llfields`: extra llfields
        // (the virtualizable token as `llmemory.GCREF`) precede the instance
        // attrs (`llfields + myllfields`).
        if let Some(vable) = self.virtualizable.borrow().as_ref() {
            myllfields.splice(0..0, vable.setup_repr_llfields());
        }
        // `_parse_field_list` reads `self.fields`. Store them before
        // `MkStruct` so the `virtualizable_accessor` hint can name the
        // redirected fields the way `accessor.fields` does after setup.
        *self.fields.borrow_mut() = fields.clone();
        let struct_hints = self.virtualizable_accessor_hint()?;
        let name = classdef_rc.borrow().name.clone();
        let mut struct_fields = Vec::with_capacity(1 + myllfields.len());
        struct_fields.push(("super".into(), rbase.object_type().clone()));
        struct_fields.extend(myllfields);
        let body = match self.gcflavor {
            Flavor::Gc => Struct::gc_rtti_with_hints(&name, struct_fields, struct_hints),
            Flavor::Raw => Struct::with_hints(&name, struct_fields, struct_hints),
        };
        let LowLevelType::ForwardReference(fwd) = &self.object_type else {
            return Err(TyperError::message(
                "InstanceRepr.object_type must be LowLevelType::ForwardReference \
                 when classdef is not None",
            ));
        };
        fwd.r#become(LowLevelType::Struct(Box::new(body)))
            .map_err(TyperError::message)?;

        // rclass.py:555-558 — `allinstancefields.update(
        // self.rbase.allinstancefields); allinstancefields.update(
        // fields); self.fields = fields;
        // self.allinstancefields = allinstancefields`.
        for (k, v) in rbase.allinstancefields.borrow().iter() {
            allinstancefields.insert(k.clone(), v.clone());
        }
        for (k, v) in fields.iter() {
            allinstancefields.insert(k.clone(), v.clone());
        }
        *self.fields.borrow_mut() = fields;
        *self.allinstancefields.borrow_mut() = allinstancefields;
        self.finish_virtualizable_setup()?;
        Ok(())
    }
}

// rclass.py getclassrepr, 91-119, 439-440 — module-level accessors.

/// RPython `get_type_repr(rtyper)` (`rclass.py`).
///
/// ```python
/// def get_type_repr(rtyper):
///     return rtyper.rootclass_repr
/// ```
pub fn get_type_repr(rtyper: &RPythonTyper) -> Result<Arc<dyn Repr>, TyperError> {
    rtyper
        .rootclass_repr
        .borrow()
        .clone()
        .map(|r| r as Arc<dyn Repr>)
        .ok_or_else(|| {
            TyperError::message(
                "rtyper.rootclass_repr is not set — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })
}

/// RPython `getclassrepr(rtyper, classdef)` (`rclass.py`).
///
/// ```python
/// def getclassrepr(rtyper, classdef):
///     if classdef is None:
///         return rtyper.rootclass_repr
///     result = classdef.repr
///     if result is None:
///         result = classdef.repr = ClassRepr(rtyper, classdef)
///         rtyper.add_pendingsetup(result)
///     return result
/// ```
///
/// The `classdef != None` arm caches through `ClassDef.repr`, matching
/// upstream's per-classdef storage.
pub fn getclassrepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
) -> Result<Arc<dyn Repr>, TyperError> {
    Ok(getclassrepr_arc(rtyper, classdef)?.as_repr())
}

/// Internal-recursion companion to [`getclassrepr`]: returns the
/// typed [`ClassReprArc`] enum so [`ClassRepr::_setup_repr`] can read
/// `rbase.vtable_type` directly without upcasting through
/// `Arc<dyn Repr>`.
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
pub(crate) fn getclassrepr_arc(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
) -> Result<ClassReprArc, TyperError> {
    let Some(classdef_rc) = classdef else {
        // rclass.py:68-69 — `if classdef is None: return rtyper.rootclass_repr`.
        let root = rtyper.rootclass_repr.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "rtyper.rootclass_repr is not set — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })?;
        return Ok(ClassReprArc::Root(root));
    };
    // rclass.py:70-73 — `classdef.repr` cache, fresh-build path.
    if let Some(existing) = classdef_rc.borrow().repr.clone() {
        return Ok(ClassReprArc::Inst(existing));
    }
    let repr = Arc::new(ClassRepr::new(rtyper, classdef_rc));
    classdef_rc.borrow_mut().repr = Some(repr.clone());
    rtyper.add_pendingsetup(repr.clone() as Arc<dyn Repr>);
    Ok(ClassReprArc::Inst(repr))
}

/// RPython `getinstancerepr(rtyper, classdef, default_flavor='gc')`
/// (`rclass.py`).
///
/// ```python
/// def getinstancerepr(rtyper, classdef, default_flavor='gc'):
///     if classdef is None:
///         flavor = default_flavor
///     else:
///         flavor = getgcflavor(classdef)
///     try:
///         result = rtyper.instance_reprs[classdef, flavor]
///     except KeyError:
///         result = buildinstancerepr(rtyper, classdef, gcflavor=flavor)
///         rtyper.instance_reprs[classdef, flavor] = result
///         rtyper.add_pendingsetup(result)
///     return result
/// ```
///
/// RPython `externalvsinternal(rtyper, item_repr, gcref=False)`
/// (rmodel.py).
///
/// ```python
/// def externalvsinternal(rtyper, item_repr, gcref=False):
///     ...
///     if (gcref and isinstance(item_repr.lowleveltype, Ptr) and
///             item_repr.lowleveltype.TO._gckind == 'gc'):
///         return item_repr, rgcref.GCRefRepr.make(item_repr, rtyper.gcrefreprcache)
///     if (isinstance(item_repr, rclass.InstanceRepr) and
///         getattr(item_repr, 'gcflavor', 'gc') == 'gc'):
///         return item_repr, rclass.getinstancerepr(rtyper, None)
///     else:
///         return item_repr, item_repr
/// ```
///
/// Returns `(external_repr, internal_repr)`. Used by container reprs
/// (TupleRepr / ListRepr / DictRepr) to store GC `InstanceRepr`s as
/// the root `InstanceRepr` (gcflavor='gc', classdef=None) internally
/// while preserving the concrete external repr at the surface.
///
/// Pyre placement deviation: upstream lives in `rmodel.py` and lazy-
/// imports `rclass`; the Rust port flips that — InstanceRepr lives in
/// `rclass.rs` and `rmodel.rs` cannot import `rclass` without
/// introducing a module-level cycle. Callers in `rtuple.rs` /
/// follow-on `rlist.rs` import this directly from `rclass`.
///
/// `gcref=True` arm (rmodel.py) routes through
/// `lltypesystem.rgcref.GCRefRepr.make`, matching upstream's generic
/// GCREF storage for GC pointer items in containers.
#[expect(
    clippy::type_complexity,
    reason = "This is the literal nested tuple/list/dict/callable shape at an RPython parity boundary; a wrapper would change structural ownership, while a one-use alias would conceal the audited upstream shape"
)]
pub(crate) fn externalvsinternal(
    rtyper: &Rc<RPythonTyper>,
    item_repr: Arc<dyn Repr>,
    gcref: bool,
) -> Result<(Arc<dyn Repr>, Arc<dyn Repr>), TyperError> {
    if gcref
        && let LowLevelType::Ptr(ptr) = item_repr.lowleveltype()
        && ptr._gckind() == crate::translator::rtyper::lltypesystem::lltype::GcKind::Gc
    {
        let internal = crate::translator::rtyper::lltypesystem::rgcref::GCRefRepr::make(
            item_repr.clone(),
            &rtyper.gcrefreprcache,
        );
        let external = item_repr;
        return Ok((external, internal as Arc<dyn Repr>));
    }
    let any_r: &dyn std::any::Any = item_repr.as_ref();
    if let Some(inst) = any_r.downcast_ref::<InstanceRepr>()
        && inst.gcflavor() == Flavor::Gc
    {
        let internal = getinstancerepr(rtyper, None, Flavor::Gc)?;
        let external = item_repr.clone();
        return Ok((external, internal as Arc<dyn Repr>));
    }
    let external = item_repr.clone();
    Ok((external, item_repr))
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "Arc preserves shared runtime descriptor/JitCode identity while non-Send translator payload remains confined to the single-threaded build phase"
)]
pub fn getinstancerepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    default_flavor: Flavor,
) -> Result<Arc<InstanceRepr>, TyperError> {
    let flavor = match classdef {
        None => default_flavor,
        Some(classdef_rc) => getgcflavor(classdef_rc)?,
    };
    let classdef_key = classdef.map(ClassDefKey::from_classdef);
    let key = (classdef_key, flavor);
    if let Some(existing) = rtyper.instance_reprs.borrow().get(&key) {
        return Ok(existing.clone());
    }
    let result = Arc::new(buildinstancerepr(rtyper, classdef, flavor)?);
    rtyper
        .instance_reprs
        .borrow_mut()
        .insert(key, result.clone());
    rtyper.add_pendingsetup(result.clone());
    Ok(result)
}

/// RPython `rtype_new_instance(rtyper, classdef, llops,
/// classcallhop=None, nonmovable=False)` (rclass.py):
///
/// ```python
/// def rtype_new_instance(rtyper, classdef, llops, classcallhop=None,
///                        nonmovable=False):
///     rinstance = getinstancerepr(rtyper, classdef)
///     return rinstance.new_instance(llops, classcallhop, nonmovable=nonmovable)
/// ```
pub fn rtype_new_instance(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    llops: &mut LowLevelOpList,
    classcallhop: Option<&crate::translator::rtyper::rtyper::HighLevelOp>,
    nonmovable: bool,
) -> Result<Hlvalue, TyperError> {
    let rinstance = getinstancerepr(rtyper, classdef, Flavor::Gc)?;
    Repr::setup(rinstance.as_ref() as &dyn Repr)?;
    rinstance.new_instance(llops, classcallhop, nonmovable)
}

/// RPython `InstanceRepr.has_special_memory_pressure(self, tp)`
/// (rclass.py):
///
/// ```python
/// def has_special_memory_pressure(self, tp):
///     if 'special_memory_pressure' in tp._flds:
///         return True
///     if 'super' in tp._flds:
///         return self.has_special_memory_pressure(tp._flds['super'])
///     return False
/// ```
fn has_special_memory_pressure(tp: &LowLevelType) -> bool {
    let LowLevelType::Struct(s) = tp else {
        return false;
    };
    if s._flds.get("special_memory_pressure").is_some() {
        return true;
    }
    match s._flds.get("super") {
        Some(sup) => has_special_memory_pressure(sup),
        None => false,
    }
}

/// RPython `pairtype(InstanceRepr, InstanceRepr).convert_from_to((r_ins1,
/// r_ins2), v, llops)` (rclass.py):
///
/// ```python
/// def convert_from_to((r_ins1, r_ins2), v, llops):
///     # which is a subclass of which?
///     if r_ins1.classdef is None or r_ins2.classdef is None:
///         basedef = None
///     else:
///         basedef = r_ins1.classdef.commonbase(r_ins2.classdef)
///     if basedef == r_ins2.classdef:
///         # r_ins1 is an instance of the subclass: converting to parent
///         v = llops.genop('cast_pointer', [v],
///                         resulttype=r_ins2.lowleveltype)
///         return v
///     elif basedef == r_ins1.classdef:
///         # r_ins2 is an instance of the subclass: potentially unsafe
///         # casting, but we do it anyway (e.g. the annotator produces
///         # such casts after a successful isinstance() check)
///         v = llops.genop('cast_pointer', [v],
///                         resulttype=r_ins2.lowleveltype)
///         return v
///     else:
///         return NotImplemented
/// ```
pub(super) fn pair_instance_instance_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    let r_ins1 = (r_from as &dyn std::any::Any)
        .downcast_ref::<InstanceRepr>()
        .ok_or_else(|| {
            TyperError::message(
                "pair_instance_instance_convert_from_to: r_from is not InstanceRepr",
            )
        })?;
    let r_ins2 = (r_to as &dyn std::any::Any)
        .downcast_ref::<InstanceRepr>()
        .ok_or_else(|| {
            TyperError::message("pair_instance_instance_convert_from_to: r_to is not InstanceRepr")
        })?;
    let basedef = match (&r_ins1.classdef, &r_ins2.classdef) {
        (Some(c1), Some(c2)) => ClassDef::commonbase(c1, c2),
        _ => None,
    };
    let classdef_eq =
        |a: &Option<Rc<RefCell<ClassDef>>>, b: &Option<Rc<RefCell<ClassDef>>>| match (a, b) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
    // Upstream's two arms (upcast to parent / downcast after a
    // successful isinstance check) both emit the same cast_pointer.
    if classdef_eq(&basedef, &r_ins2.classdef) || classdef_eq(&basedef, &r_ins1.classdef) {
        return Ok(llops
            .genop(
                "cast_pointer",
                vec![v.clone()],
                GenopResult::LLType(r_ins2.lowleveltype.clone()),
            )
            .map(Hlvalue::Variable));
    }
    Ok(None)
}

/// The low-level side of the annotator's `union(SomeInstance(None),
/// SomePtr)` (`model.rs`): a Gc-opaque pointer merged into the
/// classdef-less top instance.  Both annotations denote one type-erased GC
/// reference, the role `llmemory.GCREF` (`Ptr(GcOpaqueType)`) plays
/// upstream, so the conversion is the GCREF edge
/// `pairtype(GCRefRepr, Repr).convert_from_to` (`rgcref.py`):
/// `cast_opaque_ptr` to the instance pointer.  Any other pointer, or a
/// classdef-bearing destination, is outside that union and stays
/// `NotImplemented`.
pub(super) fn pair_opaque_ptr_instance_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    use crate::translator::rtyper::lltypesystem::lltype::{GcKind, PtrTarget};
    let LowLevelType::Ptr(ptr) = r_from.lowleveltype() else {
        return Ok(None);
    };
    let PtrTarget::Opaque(opaque) = &ptr.TO else {
        return Ok(None);
    };
    if opaque._gckind != GcKind::Gc {
        return Ok(None);
    }
    let Some(r_ins) = (r_to as &dyn std::any::Any).downcast_ref::<InstanceRepr>() else {
        return Ok(None);
    };
    if r_ins.classdef.is_some() {
        return Ok(None);
    }
    super::lltypesystem::rgcref::pair_gcref_repr_convert_from_to(r_from, r_to, v, llops)
}

/// RPython `pairtype(InstanceRepr, InstanceRepr).rtype_is_`
/// (rclass.py):
///
/// ```python
/// def rtype_is_((r_ins1, r_ins2), hop):
///     if r_ins1.gcflavor != r_ins2.gcflavor:
///         # obscure logic, the is can be true only if both are None
///         v_ins1, v_ins2 = hop.inputargs(
///             r_ins1.common_repr(), r_ins2.common_repr())
///         return hop.gendirectcall(ll_both_none, v_ins1, v_ins2)
///     if r_ins1.classdef is None or r_ins2.classdef is None:
///         basedef = None
///     else:
///         basedef = r_ins1.classdef.commonbase(r_ins2.classdef)
///     r_ins = getinstancerepr(r_ins1.rtyper, basedef, r_ins1.gcflavor)
///     return pairtype(Repr, Repr).rtype_is_(pair(r_ins, r_ins), hop)
/// ```
///
/// Both sides convert to the common-base instance repr (the upcast
/// `cast_pointer` comes from [`pair_instance_instance_convert_from_to`]
/// at `inputargs` time) and the generic pointer-identity arm emits
/// `ptr_eq`. Mixed gcflavors route through the `ll_both_none` helper
/// (`not ins1 and not ins2`, rclass.py).
pub(super) fn pair_instance_instance_rtype_is_(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    let Some(r_ins1) = (r1 as &dyn std::any::Any).downcast_ref::<InstanceRepr>() else {
        return Ok(None);
    };
    let Some(r_ins2) = (r2 as &dyn std::any::Any).downcast_ref::<InstanceRepr>() else {
        return Ok(None);
    };
    if r_ins1.gcflavor != r_ins2.gcflavor {
        // "obscure logic, the is can be true only if both are None"
        // (rclass.py:1058-1062): each side converts to its own
        // flavor's common repr, then `ll_both_none` tests both for
        // null.
        use crate::translator::rtyper::rtyper::ConvertedTo;
        let common1 = r_ins1.common_repr()?;
        let common2 = r_ins2.common_repr()?;
        let v_args = hop.inputargs(vec![
            ConvertedTo::Repr(common1.as_ref() as &dyn Repr),
            ConvertedTo::Repr(common2.as_ref()),
        ])?;
        let rtyper = r_ins1.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("pair_instance_instance_rtype_is_: rtyper weak ref expired")
        })?;
        let helper = rtyper.lowlevel_helper_function(
            "ll_both_none",
            vec![
                common1.lowleveltype().clone(),
                common2.lowleveltype().clone(),
            ],
            LowLevelType::Bool,
        )?;
        return hop.gendirectcall(&helper, v_args);
    }
    let basedef = match (&r_ins1.classdef, &r_ins2.classdef) {
        (Some(c1), Some(c2)) => ClassDef::commonbase(c1, c2),
        _ => None,
    };
    let rtyper = r_ins1.rtyper.upgrade().ok_or_else(|| {
        TyperError::message("pair_instance_instance_rtype_is_: rtyper weak ref expired")
    })?;
    let r_ins = getinstancerepr(&rtyper, basedef.as_ref(), r_ins1.gcflavor)?;
    crate::translator::rtyper::pairtype::pair_repr_repr_rtype_is_(
        r_ins.as_ref(),
        r_ins.as_ref(),
        hop,
    )
    .map(Some)
}

/// RPython `pairtype(InstanceRepr, InstanceRepr).rtype_ne`
/// (rclass.py): negate `rtype_eq` (which aliases `rtype_is_`).
pub(super) fn pair_instance_instance_rtype_ne(
    r1: &dyn Repr,
    r2: &dyn Repr,
    hop: &HighLevelOp,
) -> Result<Option<Hlvalue>, TyperError> {
    let Some(v_eq) = pair_instance_instance_rtype_is_(r1, r2, hop)? else {
        return Ok(None);
    };
    Ok(hop.genop(
        "bool_not",
        vec![v_eq],
        GenopResult::LLType(LowLevelType::Bool),
    ))
}

/// RPython `buildinstancerepr(rtyper, classdef, gcflavor='gc')`
/// (`rclass.py`).
///
pub fn buildinstancerepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    gcflavor: Flavor,
) -> Result<InstanceRepr, TyperError> {
    // rclass.py:94-103 — `classdef is None` short-circuits `unboxed =
    // []`, `virtualizable = False`; otherwise scan all subdefs for
    // `UnboxedValue`-derived classes and read `_virtualizable_`. The
    // `usetagging` gate is `len(unboxed) != 0 and
    // config.translation.taggedpointers`, so when taggedpointers is off
    // the presence of an `UnboxedValue` subclass falls through to a
    // regular `InstanceRepr`.
    let (unboxed, virtualizable) = match classdef {
        None => (Vec::new(), false),
        Some(classdef_rc) => {
            let unboxed: Vec<Rc<RefCell<ClassDef>>> = ClassDef::getallsubdefs(classdef_rc)
                .into_iter()
                .filter(|subdef| {
                    host_is_unboxed_value_subclass(&subdef.borrow().classdesc.borrow().pyobj)
                })
                .collect();
            // rclass.py `buildinstancerepr`:
            // `virtualizable = classdef.classdesc.get_param('_virtualizable_', False)`
            let virtualizable = const_truthy(&classdef_rc.borrow().classdesc.borrow().get_param(
                "_virtualizable_",
                Some(ConstValue::Bool(false)),
                true,
            ));
            (unboxed, virtualizable)
        }
    };
    // `rtyper.annotator` is weak because the annotator owns a strong
    // `Rc<RPythonTyper>` (rtyper.py `annotator.translator.rtyper` +
    // `rtyper.annotator` form a Python-side cycle; Rust breaks it by
    // keeping this edge weak). Tests that construct an RPythonTyper from
    // a non-retained `Rc<RPythonAnnotator>` observe the weak drop here.
    // Upstream `config.translation.taggedpointers` defaults to `False`,
    // so on upgrade failure we use the default — matching upstream
    // semantics for any caller that has not explicitly opted in to
    // tagged pointers.
    let taggedpointers = rtyper
        .annotator
        .upgrade()
        .map(|ann| ann.translator.config.translation.taggedpointers)
        .unwrap_or(false);
    let usetagging = !unboxed.is_empty() && taggedpointers;

    // rclass.py:105-108 — virtualizable path.
    if virtualizable {
        assert!(
            unboxed.is_empty(),
            "_virtualizable_ class must not have UnboxedValue subclasses"
        );
        assert_eq!(gcflavor, Flavor::Gc, "_virtualizable_ requires gc flavor");
        let classdef = classdef.expect("virtualizable InstanceRepr needs a classdef");
        let classdesc = classdef.borrow().classdesc.clone();
        let extra =
            super::rvirtualizable::VirtualizableInstanceRepr::from_classdesc(&classdesc.borrow())?;
        let repr = InstanceRepr::new(rtyper, Some(classdef), gcflavor);
        *repr.virtualizable.borrow_mut() = Some(extra);
        return Ok(repr);
    }
    // rclass.py:109-117 — tagged-pointer path.
    if usetagging {
        if unboxed.len() != 1 {
            return Err(TyperError::message(format!(
                "{:?} has several UnboxedValue subclasses",
                classdef.map(|cd| cd.borrow().name.clone())
            )));
        }
        assert_eq!(
            gcflavor,
            Flavor::Gc,
            "UnboxedValue tagging requires gc flavor"
        );
        return Err(TyperError::message(
            "buildinstancerepr: TaggedInstanceRepr parity requires \
             rpython/rtyper/lltypesystem/rtagged.py (not yet ported)",
        ));
    }
    // rclass.py — default `InstanceRepr`. An `UnboxedValue`
    // subclass without `config.translation.taggedpointers` falls through
    // here upstream too, so this arm owns that parity case.
    match classdef {
        None => Ok(InstanceRepr::new_rootinstance(rtyper, gcflavor)),
        Some(classdef_rc) => Ok(InstanceRepr::new(rtyper, Some(classdef_rc), gcflavor)),
    }
}

/// Cache key shape `(Option<ClassDefKey>, Flavor)` used by
/// [`RPythonTyper::instance_reprs`]. Exposed so consumers outside this
/// module can build matching queries without importing `Flavor` + the
/// field directly.
pub(crate) type InstanceReprKey = (Option<ClassDefKey>, Flavor);

// Unit tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::GcKind;

    #[test]
    fn top_level_runtime_helper_parity_surface() {
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc, nullptr,
        };

        assert_eq!(ll_inst_hash(None), 0);
        let null_object = nullptr(OBJECT.clone()).expect("nullptr object");
        assert_eq!(ll_inst_hash(Some(&null_object)), 0);

        let struct_t = LowLevelType::Struct(Box::new(Struct::gc(
            "pkg.C",
            vec![("inst_x".into(), LowLevelType::Signed)],
        )));
        let mut ptr = malloc(struct_t, None, MallocFlavor::Gc, true).expect("malloc instance");
        feedllattr(&mut ptr, "x", LowLevelValue::Signed(42)).expect("feedllattr");
        assert_eq!(
            fishllattr(&ptr, "x", None).expect("fishllattr"),
            LowLevelValue::Signed(42)
        );
        assert_eq!(
            fishllattr(&ptr, "missing", Some(LowLevelValue::Signed(7))).expect("default"),
            LowLevelValue::Signed(7)
        );
        assert_eq!(ll_inst_type(None).expect("None type"), None);

        let mut subcls = malloc(
            LowLevelType::Struct(Box::new(Struct::gc(
                "vtable",
                vec![("subclassrange_min".into(), LowLevelType::Signed)],
            ))),
            None,
            MallocFlavor::Gc,
            true,
        )
        .expect("malloc vtable");
        subcls
            .setattr("subclassrange_min", LowLevelValue::Signed(7))
            .expect("set subclassrange_min");
        assert!(ll_issubclass_const(&subcls, 3, 9).expect("inside range"));
        assert!(!ll_issubclass_const(&subcls, 8, 12).expect("outside range"));

        let err = declare_type_for_typeptr(&ptr, &OBJECT.clone()).expect_err("deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("declare_type_for_typeptr"));

        let conflict = ImmutableConflictError {
            message: "conflict".to_string(),
        };
        assert_eq!(conflict.message, "conflict");
        let _missing_marker = _missing.clone();
    }

    #[test]
    fn object_vtable_resolves_to_struct_with_subclassrange_fields() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!("OBJECT_VTABLE must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("OBJECT_VTABLE should be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("OBJECT_VTABLE must resolve to a Struct");
        };
        assert_eq!(body._name, "object_vtable");
        assert!(body._flds.get("subclassrange_min").is_some());
        assert!(body._flds.get("subclassrange_max").is_some());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("static_immutable"),
            Some(&ConstValue::Bool(true))
        );
    }

    /// `OBJECT_VTABLE.name: Ptr(rstr.STR)` (rclass.py). The `name`
    /// field is consumed by `ClassesPBCRepr.rtype_getattr('__name__')`
    /// (rpbc.py), which emits a `getfield` op against this
    /// slot. The lltype of the field must be `Ptr(STR)` resolving to
    /// the `rpy_string` GcStruct.
    /// `OBJECT_VTABLE.instantiate: Ptr(FuncType([], OBJECTPTR))`
    /// (rclass.py). Closes the cycle `OBJECT_VTABLE → OBJECTPTR
    /// → OBJECT → CLASSTYPE → OBJECT_VTABLE`. The Rust port resolves
    /// it via the `OBJECT_FAMILY` LazyLock that builds an unresolved
    /// ForwardReference first, constructs the four types, then
    /// `become()`s the ForwardReference to the full vtable Struct
    /// (mirroring upstream's mutable-ForwardReference + post-hoc
    /// `become` ordering).
    #[test]
    fn object_vtable_carries_instantiate_funcptr_returning_objectptr() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!("OBJECT_VTABLE must be ForwardReference");
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!("OBJECT_VTABLE must resolve to a Struct");
        };
        let instantiate_type = body
            ._flds
            .get("instantiate")
            .expect("instantiate field must be present (rclass.py:172)");
        let LowLevelType::Ptr(funcptr) = instantiate_type else {
            panic!("instantiate must be a Ptr, got {instantiate_type:?}");
        };
        let PtrTarget::Func(funcsig) = &funcptr.TO else {
            panic!(
                "instantiate's Ptr target must be Func, got {:?}",
                funcptr.TO
            );
        };
        assert!(
            funcsig.args.is_empty(),
            "instantiate funcptr has zero args, got {} args",
            funcsig.args.len()
        );
        // Result must be OBJECTPTR.
        assert_eq!(
            &funcsig.result, &*OBJECTPTR,
            "instantiate funcptr result must be OBJECTPTR"
        );
    }

    #[test]
    fn object_vtable_carries_name_ptr_to_str() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!();
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!();
        };
        let name_type = body
            ._flds
            .get("name")
            .expect("name field must be present (rclass.py:171)");
        let LowLevelType::Ptr(ptr) = name_type else {
            panic!("name must be a Ptr, got {name_type:?}");
        };
        // STR is a ForwardReference resolving to rpy_string.
        let PtrTarget::ForwardReference(fwd) = &ptr.TO else {
            panic!(
                "name's Ptr target must be ForwardReference (STR), got {:?}",
                ptr.TO
            );
        };
        let LowLevelType::Struct(str_body) = fwd.resolved().expect("STR must resolve") else {
            panic!("STR must resolve to a Struct");
        };
        assert_eq!(str_body._name, "rpy_string");
    }

    #[test]
    fn object_vtable_carries_rtti_ptr_to_runtime_type_info() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!();
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!();
        };
        let rtti_type = body
            ._flds
            .get("rtti")
            .expect("rtti field must be present (rclass.py:171)");
        let LowLevelType::Ptr(ptr) = rtti_type else {
            panic!("rtti must be a Ptr, got {rtti_type:?}");
        };
        let PtrTarget::Opaque(opaque) = &ptr.TO else {
            panic!("rtti must point to an opaque, got {:?}", ptr.TO);
        };
        assert_eq!(opaque.tag, "RuntimeTypeInfo");
    }

    #[test]
    fn classtype_is_ptr_to_object_vtable_forward_reference() {
        let LowLevelType::Ptr(ptr) = CLASSTYPE.clone() else {
            panic!("CLASSTYPE must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn object_is_gcstruct_with_typeptr_field_and_hints() {
        let LowLevelType::Struct(body) = OBJECT.clone() else {
            panic!("OBJECT must be Struct");
        };
        assert_eq!(body._name, "object");
        assert_eq!(body._gckind, GcKind::Gc);
        assert!(body._flds.get("typeptr").is_some());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("shouldntbenull"),
            Some(&ConstValue::Bool(true))
        );
        assert_eq!(body._hints.get("typeptr"), Some(&ConstValue::Bool(true)));
    }

    #[test]
    fn object_carries_rtti_opaque_under_runtime_type_info() {
        // rclass.py sets `rtti=True` on OBJECT so
        // `getRuntimeTypeInfo(OBJECT)` resolves. R3 consumers
        // (`fill_vtable_root`) rely on this.
        let LowLevelType::Struct(body) = OBJECT.clone() else {
            panic!("OBJECT must be Struct");
        };
        let opaque = body
            ._runtime_type_info
            .as_ref()
            .expect("OBJECT must carry an _runtime_type_info opaque");
        assert_eq!(opaque._name.as_deref(), Some("object"));
    }

    #[test]
    fn nongcobject_is_raw_struct_with_typeptr_field() {
        let LowLevelType::Struct(body) = NONGCOBJECT.clone() else {
            panic!("NONGCOBJECT must be Struct");
        };
        assert_eq!(body._name, "nongcobject");
        assert_eq!(body._gckind, GcKind::Raw);
    }

    #[test]
    fn object_by_flavor_returns_gc_and_raw_forms() {
        assert_eq!(object_by_flavor(Flavor::Gc), OBJECT.clone());
        assert_eq!(object_by_flavor(Flavor::Raw), NONGCOBJECT.clone());
    }

    #[test]
    fn root_class_repr_lowleveltype_is_classtype() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = RootClassRepr::new(&rtyper);
        assert_eq!(repr.lowleveltype(), &CLASSTYPE.clone());
        assert_eq!(repr.vtable_type(), &OBJECT_VTABLE.clone());
        assert_eq!(repr.classdef(), None);
    }

    #[test]
    fn root_class_repr_setup_is_noop_idempotent() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let repr = RootClassRepr::new(&rtyper);
        Repr::setup(&repr).expect("first setup");
        Repr::setup(&repr).expect("second setup re-enters Finished branch");
    }

    #[test]
    fn root_class_repr_init_vtable_writes_subclassrange_and_rtti() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = rtyper
            .rootclass_repr
            .borrow()
            .clone()
            .expect("rootclass_repr after initialize");
        let vtable = repr.getvtable().expect("getvtable");
        let min = vtable.getattr("subclassrange_min").unwrap();
        let max = vtable.getattr("subclassrange_max").unwrap();
        assert_eq!(min, lltype::LowLevelValue::Signed(0));
        assert_eq!(max, lltype::LowLevelValue::Signed(i64::MAX));
        // rtti slot is populated when gcflavor == Gc.
        let rtti = vtable.getattr("rtti").unwrap();
        let lltype::LowLevelValue::Ptr(rtti_ptr) = rtti else {
            panic!("rtti must be a Ptr value");
        };
        assert!(
            rtti_ptr.nonzero(),
            "rtti pointer must be live for Gc flavor"
        );
    }

    #[test]
    fn class_repr_init_vtable_writes_subclassrange_from_classdef_minmax_id() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let host = HostObject::new_class("C", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(7);
            cd.maxid = Some(11);
        }
        let r = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r.as_repr()).expect("setup classrepr");

        let ClassReprArc::Inst(inst_repr) = r else {
            panic!("expected ClassReprArc::Inst");
        };
        let vtable = inst_repr.getvtable().expect("getvtable");
        // The per-class vtable Struct is wrapped — `cast_vtable_to_typeptr`
        // already walked `super` down to the OBJECT_VTABLE level, so the
        // returned pointer carries the root subclassrange writes from
        // `fill_vtable_root`.
        let min = vtable.getattr("subclassrange_min").unwrap();
        let max = vtable.getattr("subclassrange_max").unwrap();
        assert_eq!(min, lltype::LowLevelValue::Signed(7));
        assert_eq!(max, lltype::LowLevelValue::Signed(11));
    }

    #[test]
    fn cast_vtable_to_typeptr_walks_super_chain_to_object_vtable() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let host = HostObject::new_class("C", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        {
            let mut cd = classdef.borrow_mut();
            cd.minid = Some(2);
            cd.maxid = Some(3);
        }
        let r = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r.as_repr()).expect("setup classrepr");
        let ClassReprArc::Inst(inst_repr) = r else {
            unreachable!();
        };
        // Internal pre-cast vtable still points at the per-class
        // struct (`C_vtable`); after `getvtable`'s
        // `cast_vtable_to_typeptr` it points at `OBJECT_VTABLE`.
        inst_repr.init_vtable().unwrap();
        let raw = inst_repr.vtable.borrow().clone().unwrap();
        let raw_to = LowLevelType::from(raw._TYPE.TO.clone());
        assert_ne!(
            raw_to,
            OBJECT_VTABLE.clone(),
            "raw vtable points at C_vtable"
        );

        let cast = cast_vtable_to_typeptr(raw);
        let cast_to = LowLevelType::from(cast._TYPE.TO.clone());
        assert_eq!(
            cast_to,
            OBJECT_VTABLE.clone(),
            "cast walked to OBJECT_VTABLE"
        );
    }

    #[test]
    fn root_class_repr_convert_desc_returns_vtable_constant() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        // ClassRepr.convert_desc upstream is the conversion entry for
        // SomeType-typed values (which route through `get_type_repr →
        // rtyper.rootclass_repr`). The receiver is therefore the root
        // ClassRepr whose `lowleveltype == CLASSTYPE`, satisfying the
        // `getruntime` assert. Non-root ClassRepr.convert_desc is
        // exercised only via the inherited shape but never reached
        // with a mismatched `expected_type`.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("C", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        classdef.borrow_mut().minid = Some(5);
        classdef.borrow_mut().maxid = Some(5);
        // Pre-setup the subclass ClassRepr so its vtable_type
        // ForwardReference is resolved when `getruntime → init_vtable`
        // runs from `convert_desc`. Real-world flow has this happen via
        // `RPythonTyper.specialize` → `_setup_repr` chain.
        let r_sub = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r_sub.as_repr()).unwrap();

        let root = rtyper.rootclass_repr.borrow().clone().unwrap();
        let out = root.convert_desc(&entry).expect("convert_desc(C)");
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero());
        assert_eq!(out.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
    }

    #[test]
    fn fill_vtable_root_stores_my_instantiate_graph() {
        use crate::annotator::model::SomeInstance;
        use crate::flowspace::model::GraphKey;
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, LowLevelValue};
        use crate::translator::rtyper::normalizecalls::create_instantiate_function;
        use crate::translator::rtyper::rtyper::RPythonTyper;
        use std::rc::Rc;

        let ann = crate::annotator::annrpython::RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let classdef = ClassDef::new_standalone("pkg.Inst", None);
        classdef.borrow_mut().minid = Some(3);
        classdef.borrow_mut().maxid = Some(3);
        create_instantiate_function(&ann, &classdef).expect("instantiate graph");
        let graph = classdef
            .borrow()
            .my_instantiate_graph
            .clone()
            .expect("my_instantiate_graph");
        // create_instantiate_function binds the return to SomeInstance(None).
        // Confirm that binding is the one getfunctionptr will read.
        let ret = graph.borrow().getreturnvar();
        let Hlvalue::Variable(ret_var) = &ret else {
            panic!("return var");
        };
        let bound = ret_var.annotation.borrow().as_ref().map(|s| (**s).clone());
        assert!(matches!(
            bound,
            Some(SomeValue::Instance(SomeInstance { classdef: None, .. }))
        ));

        let repr_arc = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(repr_arc.as_ref()).expect("setup");
        repr_arc.init_vtable().expect("init_vtable");
        let mut vtable = repr_arc.vtable.borrow().clone().expect("vtable");
        let slot = loop {
            if let Ok(slot) = vtable.getattr("instantiate") {
                break slot;
            }
            let LowLevelValue::Ptr(parent) = vtable.getattr("super").expect("vtable super") else {
                panic!("vtable super must be a pointer");
            };
            vtable = *parent;
        };
        let LowLevelValue::Ptr(funcptr) = slot else {
            panic!("instantiate must be a function pointer, got {slot:?}");
        };
        let Ok(_ptr_obj::Func(func)) = funcptr._obj() else {
            panic!("instantiate pointer must expose a function");
        };
        assert_eq!(func.graph, Some(GraphKey::of(&graph).as_usize()));
        assert!(func.TYPE.args.is_empty());
        assert_eq!(&func.TYPE.result, &*OBJECTPTR);
    }

    #[test]
    fn root_class_repr_convert_const_routes_through_bookkeeper() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("D", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(class_rc) = &entry else {
            unreachable!();
        };
        let classdef = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(class_rc).unwrap();
        classdef.borrow_mut().minid = Some(13);
        classdef.borrow_mut().maxid = Some(13);
        let r_sub = getclassrepr_arc(&rtyper, Some(&classdef)).unwrap();
        Repr::setup(&*r_sub.as_repr()).unwrap();
        let root = rtyper.rootclass_repr.borrow().clone().unwrap();
        let out = Repr::convert_const(root.as_ref(), &ConstValue::HostObject(host))
            .expect("convert_const HostObject(class)");
        let ConstValue::LLPtr(p) = &out.value else {
            panic!("expected LLPtr, got {:?}", out.value);
        };
        assert!(p.nonzero());

        // Non-class HostObject is rejected upfront.
        let module_host = HostObject::new_module("not_a_class_mod");
        let err =
            Repr::convert_const(root.as_ref(), &ConstValue::HostObject(module_host)).unwrap_err();
        assert!(err.to_string().contains("not a class"));
    }

    #[test]
    fn class_repr_convert_desc_rejects_unrelated_class() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        // Two unrelated classes (no common ancestor in HostObject mro
        // beyond `object`). The receiver `inst_a` rejects converting
        // for desc(B) because B is not a subclass of A.
        let host_a = HostObject::new_class("A", vec![]);
        let host_b = HostObject::new_class("B", vec![]);
        let entry_a = ann.bookkeeper.getdesc(&host_a).unwrap();
        let entry_b = ann.bookkeeper.getdesc(&host_b).unwrap();
        let crate::annotator::description::DescEntry::Class(rc_a) = &entry_a else {
            unreachable!();
        };
        let cd_a = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc_a).unwrap();
        cd_a.borrow_mut().minid = Some(1);
        cd_a.borrow_mut().maxid = Some(1);

        let r_a = getclassrepr_arc(&rtyper, Some(&cd_a)).unwrap();
        Repr::setup(&*r_a.as_repr()).unwrap();
        let ClassReprArc::Inst(inst_a) = r_a else {
            unreachable!();
        };
        // Subclass check fires before the (assert-violating) getruntime
        // call, so this exercises just the commonbase-rejection branch.
        let err = inst_a.convert_desc(&entry_b).unwrap_err();
        assert!(err.to_string().contains("not a subclass"), "{err}");
    }

    #[test]
    fn class_repr_fromtypeptr_emits_cast_pointer_with_self_lowleveltype() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("E", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(1);
        cd.borrow_mut().maxid = Some(1);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let vcls = Variable::new();
        vcls.set_concretetype(Some(CLASSTYPE.clone()));
        let v = inst
            .fromtypeptr(Hlvalue::Variable(vcls), &mut llops)
            .expect("fromtypeptr");
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "cast_pointer");
        assert_eq!(v.concretetype().as_ref(), Some(inst.lowleveltype()));
    }

    #[test]
    fn class_repr_getclsfield_traverses_rbase_until_root_then_errors() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("F", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(1);
        cd.borrow_mut().maxid = Some(1);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let vcls = Variable::new();
        vcls.set_concretetype(Some(CLASSTYPE.clone()));
        // No clsfield named "missing_attr" anywhere in the chain →
        // routes through F → rbase (RootClassRepr) which terminates
        // with a MissingRTypeAttribute-style structured error.
        let err = inst
            .getclsfield(Hlvalue::Variable(vcls), "missing_attr", &mut llops)
            .unwrap_err();
        assert!(err.to_string().contains("MissingRTypeAttribute"));
    }

    #[test]
    fn class_repr_getpbcfield_internal_error_when_field_unknown() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("G", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(1);
        cd.borrow_mut().maxid = Some(1);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let vcls = Variable::new();
        vcls.set_concretetype(Some(CLASSTYPE.clone()));
        let err = inst
            .getpbcfield(Hlvalue::Variable(vcls), 0, "missing_pbc_attr", &mut llops)
            .unwrap_err();
        assert!(err.to_string().contains("missing PBC field"));
    }

    #[test]
    fn class_repr_rtype_issubtype_const_branch_emits_direct_call_to_ll_issubclass_const() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{HostObject, SpaceOperation};
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::HighLevelOp;
        use std::cell::RefCell as StdRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("Z", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(2);
        cd.borrow_mut().maxid = Some(2);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        // Build a HighLevelOp where v_cls2 is a HostObject-typed
        // class Constant. `inputargs(class_repr)` routes through
        // RootClassRepr.convert_const → convert_desc → getruntime,
        // producing a CLASSTYPE-typed Constant whose value is the
        // vtable `_ptr`. The const-branch probe then reads
        // `subclassrange_min/max` off that vtable.
        let v_cls1 = Variable::new();
        v_cls1.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls1_h = Hlvalue::Variable(v_cls1);

        let v_cls2_h = Hlvalue::Constant(Constant::new(ConstValue::HostObject(host)));

        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::IsSubtype.opname(),
            vec![v_cls1_h.clone(), v_cls2_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_cls1_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        let class_repr_arc: Arc<dyn Repr> =
            rtyper.rootclass_repr.borrow().clone().unwrap() as Arc<dyn Repr>;
        hop.args_r.borrow_mut().push(Some(class_repr_arc.clone()));
        hop.args_v.borrow_mut().push(v_cls2_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc));

        let _ = inst.rtype_issubtype(&hop).expect("rtype_issubtype");
        let ops = hop.llops.borrow();
        let dc = ops
            .ops
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("direct_call to ll_issubclass_const helper expected");
        // direct_call args: [funcptr, v_cls1, c_min, c_max]
        assert_eq!(dc.args.len(), 4);
        let Hlvalue::Constant(c_min) = &dc.args[2] else {
            panic!("c_min must be a Constant");
        };
        let Hlvalue::Constant(c_max) = &dc.args[3] else {
            panic!("c_max must be a Constant");
        };
        assert_eq!(
            c_min.value,
            ConstValue::InheritanceId {
                cdef_id: None,
                is_max: false,
                value: 2,
            }
        );
        assert_eq!(
            c_max.value,
            ConstValue::InheritanceId {
                cdef_id: None,
                is_max: true,
                value: 2,
            }
        );
    }

    #[test]
    fn class_repr_rtype_issubtype_var_branch_emits_direct_call_to_ll_issubclass() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::HighLevelOp;
        use std::cell::RefCell as StdRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let root = rtyper.rootclass_repr.borrow().clone().unwrap();
        // Use rootclass to keep the example simple — the variable
        // branch doesn't need a subclass classdef.

        let v_cls1 = Variable::new();
        v_cls1.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls2 = Variable::new();
        v_cls2.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls1_h = Hlvalue::Variable(v_cls1);
        let v_cls2_h = Hlvalue::Variable(v_cls2);
        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::IsSubtype.opname(),
            vec![v_cls1_h.clone(), v_cls2_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_cls1_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        let class_repr_arc: Arc<dyn Repr> = root.clone() as Arc<dyn Repr>;
        hop.args_r.borrow_mut().push(Some(class_repr_arc.clone()));
        hop.args_v.borrow_mut().push(v_cls2_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc));

        // Build a synthetic ClassRepr instance just so we can invoke
        // rtype_issubtype. The receiver only needs `rtyper` for
        // helper-function lookup; classdef.minid/maxid are not read in
        // the variable branch.
        let host = crate::flowspace::model::HostObject::new_class("V", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(0);
        cd.borrow_mut().maxid = Some(0);
        let r = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*r.as_repr()).unwrap();
        let ClassReprArc::Inst(inst) = r else {
            unreachable!();
        };

        let _ = inst.rtype_issubtype(&hop).expect("rtype_issubtype");
        let ops = hop.llops.borrow();
        let dc = ops
            .ops
            .iter()
            .find(|op| op.opname == "direct_call")
            .expect("direct_call expected");
        // funcptr + 2 args.
        assert_eq!(dc.args.len(), 3);
    }

    /// `RPythonTyper::translate_operation("issubtype")` dispatches
    /// `r.rtype_issubtype(hop)` via the `Repr` trait. Without the
    /// trait override on `RootClassRepr` (which is what `SomeType`
    /// values rtype to) this would fall to the trait default
    /// `missing_rtype_operation`. The test calls
    /// `translate_operation` end-to-end to pin the trait dispatch
    /// path — not just the inherent method.
    #[test]
    fn translate_operation_issubtype_routes_through_root_class_repr_trait_override() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::SpaceOperation;
        use crate::flowspace::operation::OpKind;
        use crate::translator::rtyper::rtyper::HighLevelOp;
        use std::cell::RefCell as StdRef;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let root = rtyper.rootclass_repr.borrow().clone().unwrap();

        let v_cls1 = Variable::new();
        v_cls1.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls2 = Variable::new();
        v_cls2.set_concretetype(Some(CLASSTYPE.clone()));
        let v_cls1_h = Hlvalue::Variable(v_cls1);
        let v_cls2_h = Hlvalue::Variable(v_cls2);
        let result_var = Variable::new();
        result_var.set_concretetype(Some(LowLevelType::Bool));
        let spaceop = SpaceOperation::new(
            OpKind::IsSubtype.opname(),
            vec![v_cls1_h.clone(), v_cls2_h.clone()],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        let class_repr_arc: Arc<dyn Repr> = root as Arc<dyn Repr>;
        hop.args_v.borrow_mut().push(v_cls1_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc.clone()));
        hop.args_v.borrow_mut().push(v_cls2_h);
        hop.args_s
            .borrow_mut()
            .push(crate::annotator::model::SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(class_repr_arc));

        rtyper
            .translate_operation(&hop)
            .expect("translate_operation issubtype must dispatch through RootClassRepr");
        let ops = hop.llops.borrow();
        assert!(
            ops.ops.iter().any(|op| op.opname == "direct_call"),
            "translate_operation issubtype must emit a direct_call (ll_issubclass), \
             got {:?}",
            ops.ops.iter().map(|op| &op.opname).collect::<Vec<_>>()
        );
    }

    #[test]
    fn root_class_repr_getruntime_returns_vtable() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = rtyper.rootclass_repr.borrow().clone().unwrap();
        let v1 = repr.getruntime(&CLASSTYPE.clone()).unwrap();
        let v2 = repr.getruntime(&CLASSTYPE.clone()).unwrap();
        // Lazy cache: two calls return the same _ptr identity.
        assert_eq!(v1._hashable_identity(), v2._hashable_identity());
        let err = repr.getruntime(&LowLevelType::Signed).unwrap_err();
        assert!(err.to_string().contains("expected CLASSTYPE"));
    }

    #[test]
    fn instance_repr_none_classdef_carries_gc_object_type() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        assert!(repr.classdef().is_none());
        assert_eq!(repr.gcflavor(), Flavor::Gc);
        assert_eq!(repr.object_type(), &OBJECT.clone());
        let LowLevelType::Ptr(ptr) = repr.lowleveltype().clone() else {
            panic!("InstanceRepr.lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::Struct(_)));
    }

    #[test]
    fn instance_repr_none_classdef_setup_ok() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        Repr::setup(&repr).expect("setup classdef=None path");
    }

    #[test]
    fn generic_enum_variant_repr_contains_payload_after_constructor_setattr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::front::StructFieldRegistry;
        use std::collections::HashMap;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let root = "option::Option<Vec<u8>>";
        let variant = "option::Option<Vec<u8>>::Some";
        let mut fields = StructFieldRegistry::default();
        fields.fields.insert(
            "option::Option".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        fields.fields.insert(
            "option::Option::Some".to_string(),
            vec![("__pos_0".to_string(), "??T".to_string())],
        );
        fields.fields.insert(
            variant.to_string(),
            vec![("__pos_0".to_string(), "Vec<u8>".to_string())],
        );
        ann.bookkeeper.set_struct_fields(Rc::new(fields));

        let mut variants = HashMap::new();
        variants.insert(
            root.to_string(),
            HashMap::from([(0, "Some".to_string()), (1, "None".to_string())]),
        );
        ann.bookkeeper
            .set_enum_variant_by_discriminant(Rc::new(variants));

        // CallRegistry::ensure_session projects every registry root first,
        // then pre-mints the enum subtree before any subject is rtyped.
        for name in ann.bookkeeper.struct_root_names() {
            ann.bookkeeper
                .getuniqueclassdef_for_struct_root(&name)
                .expect("project struct root");
        }
        ann.bookkeeper.pre_register_enum_variant_classes();

        let classdef = ann
            .bookkeeper
            .getuniqueclassdef_for_enum_variant(root, "Some")
            .expect("resolve preminted Some classdef");
        let payload = classdef.borrow().attrs["__pos_0"].s_value.clone();
        classdef
            .borrow_mut()
            .attrs
            .get_mut("__pos_0")
            .expect("variant payload")
            .modified(Some(&classdef))
            .expect("constructor setattr marks payload mutable");
        crate::annotator::classdesc::ClassDef::generalize_attr(&classdef, "__pos_0", Some(payload))
            .expect("constructor setattr marks the payload as an instance field");
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("build preminted Some repr");
        Repr::setup(repr.as_ref()).expect("setup preminted Some repr");
        assert!(
            repr.fields().contains_key("__pos_0"),
            "variant payload must be present when InstanceRepr is first set up"
        );
    }

    #[test]
    fn instance_repr_pyerror_to_exc_object_emits_residual_direct_call() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInstance, SomeObject, SomeValue};
        use crate::flowspace::model::{HostObject, SpaceOperation};
        use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
        use std::cell::RefCell as StdRef;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("pyre_interpreter.error.PyError", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let repr = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).unwrap();
        Repr::setup(repr.as_ref()).unwrap();

        let v_self = Variable::new();
        v_self.set_concretetype(Some(repr.lowleveltype().clone()));
        let v_self_h = Hlvalue::Variable(v_self);
        let v_result = Variable::new();
        v_result.set_concretetype(Some(OBJECTPTR.clone()));
        let spaceop = SpaceOperation::new(
            "simple_call",
            vec![v_self_h.clone()],
            Hlvalue::Variable(v_result),
        );
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_self_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Instance(SomeInstance {
                base: SomeObject::default(),
                classdef: Some(cd),
                can_be_none: false,
                flags: Default::default(),
            }));
        hop.args_r.borrow_mut().push(Some(repr.clone()));
        let r_result = getinstancerepr(&rtyper, None, Flavor::Gc).unwrap() as Arc<dyn Repr>;
        *hop.r_result.borrow_mut() = Some(r_result);

        let out = repr
            .rtype_method("to_exc_object", &hop)
            .expect("PyError.to_exc_object rtypes")
            .expect("direct_call returns PyObjectRef");
        assert!(matches!(out, Hlvalue::Variable(_)));

        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        let op = &ops.ops[0];
        assert_eq!(op.opname, "direct_call");
        assert_eq!(op.args.len(), 2);
        let Hlvalue::Constant(c_func) = &op.args[0] else {
            panic!("direct_call arg0 must be a function pointer constant");
        };
        let ConstValue::LLPtr(ptr) = &c_func.value else {
            panic!("direct_call arg0 must carry LLPtr");
        };
        let Ok(_ptr_obj::Func(func)) = ptr._obj() else {
            panic!("direct_call arg0 must point to a function");
        };
        assert_eq!(func._name, "PyError::to_exc_object");
    }

    // R2-A — ClassRepr scaffold + getclassrepr(classdef != None).

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper
    }

    #[test]
    #[allow(non_snake_case)]
    fn instance_repr_none_classdef_populates___class___field() {
        let rtyper = fresh_rtyper();
        let repr = getinstancerepr(&rtyper, None, Flavor::Gc).expect("root instance repr");
        Repr::setup(repr.as_ref()).expect("setup root instance repr");

        let fields = repr.fields();
        let (mangled, field_repr) = fields
            .get("__class__")
            .expect("root instance must expose __class__");
        assert_eq!(mangled, "typeptr");
        assert_eq!(field_repr.lowleveltype(), &CLASSTYPE.clone());
        assert!(repr.allinstancefields().contains_key("__class__"));
    }

    #[test]
    fn instance_repr_rtype_type_nonnull_reads_class_field() {
        use crate::annotator::model::{SomeInstance, SomeObject, SomeTypeOf, SomeValue};
        use crate::flowspace::model::SpaceOperation;
        use std::cell::RefCell as StdRef;

        let rtyper = fresh_rtyper();
        let repr = getinstancerepr(&rtyper, None, Flavor::Gc).expect("root instance repr");
        Repr::setup(repr.as_ref()).expect("setup root instance repr");

        let v_obj = Variable::new();
        v_obj.set_concretetype(Some(repr.lowleveltype().clone()));
        let v_obj_h = Hlvalue::Variable(v_obj.clone());
        let v_result = Variable::new();
        let spaceop =
            SpaceOperation::new("type", vec![v_obj_h.clone()], Hlvalue::Variable(v_result));
        let llops = Rc::new(StdRef::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper.clone(), spaceop, Vec::new(), llops);
        hop.args_v.borrow_mut().push(v_obj_h);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Instance(SomeInstance {
                base: SomeObject::default(),
                classdef: None,
                can_be_none: false,
                flags: Default::default(),
            }));
        hop.args_r.borrow_mut().push(Some(repr.clone()));
        *hop.s_result.borrow_mut() = Some(SomeValue::TypeOf(SomeTypeOf::new(vec![Rc::new(v_obj)])));
        *hop.r_result.borrow_mut() = Some(get_type_repr(&rtyper).expect("type repr"));

        let out = repr
            .rtype_type(&hop)
            .expect("nonnull instance type() must rtype")
            .expect("type() returns a class pointer");
        assert!(matches!(out, Hlvalue::Variable(_)));
        let ops = hop.llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getfield");
        let Hlvalue::Variable(result) = &ops.ops[0].result else {
            panic!("getfield result must be a Variable")
        };
        assert_eq!(result.concretetype(), Some(CLASSTYPE.clone()));
    }

    #[test]
    fn instance_repr_get_reusable_prebuilt_instance_caches_and_writes_typeptr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        let host = HostObject::new_class("ExcClass", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        cd.borrow_mut().minid = Some(11);
        cd.borrow_mut().maxid = Some(11);

        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        // Setup the matching ClassRepr too so its vtable_type
        // ForwardReference is resolved when `initialize_prebuilt_data`
        // walks down to root and calls `getvtable`.
        let rclass_arc = getclassrepr_arc(&rtyper, Some(&cd)).unwrap();
        Repr::setup(&*rclass_arc.as_repr()).unwrap();

        let p1 = inst.get_reusable_prebuilt_instance().expect("first call");
        let p2 = inst.get_reusable_prebuilt_instance().expect("second call");
        // upstream `try: return self._reusable_prebuilt_instance`
        // returns the cached identity on subsequent calls.
        assert_eq!(p1._hashable_identity(), p2._hashable_identity());
        assert!(p1.nonzero(), "prebuilt instance must be live");

        // The recursive initialize_prebuilt_data walked down to the
        // OBJECT-level substruct via `setattr_at_path(["super"],
        // "typeptr", ...)` so the root super substruct now carries the
        // vtable pointer. Verify by fetching the parent allocation's
        // `super._fields` directly.
        let _ptr_obj_actual = p1
            ._obj()
            .expect("prebuilt _ptr must expose underlying object");
        let lltype::_ptr_obj::Struct(outer) = _ptr_obj_actual else {
            panic!("prebuilt instance must be a Struct");
        };
        let super_field = outer
            ._getattr("super")
            .expect("ExcClass instance has 'super' field");
        let lltype::LowLevelValue::Struct(super_struct) = super_field else {
            panic!("super field must be a Struct value");
        };
        let typeptr_val = super_struct
            ._getattr("typeptr")
            .expect("OBJECT.typeptr field present");
        let lltype::LowLevelValue::Ptr(typeptr) = typeptr_val else {
            panic!("typeptr must be a Ptr value");
        };
        assert!(typeptr.nonzero(), "typeptr must point at the class vtable");
    }

    #[test]
    fn instance_repr_setup_terminates_on_cyclic_struct_graph() {
        // Localizes the route-b stack overflow at the RTYPER
        // layer. `getuniqueclassdef_for_struct_root` (transitive
        // registration) is already proven cycle-safe (bookkeeper test
        // `getuniqueclassdef_for_struct_root_terminates_on_cyclic_graph`).
        // This test exercises the DOWNSTREAM consumer: building + setting
        // up the `InstanceRepr` for a classfull instance whose attrs form
        // a cycle (`FixedObjectArray._items: [PyObjectRef;0]` →
        // `PyObjectRef.typ: W_TypeObject` → `W_TypeObject.base:
        // PyObjectRef`). If `_setup_repr` / `getrepr` / `call_all_setups`
        // recurse without bound on the instance graph, this overflows;
        // RPython's deferred-setup worklist + `None` recursion sentinel
        // keep it bounded.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::front::StructFieldRegistry;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");

        // Pure-instance cycle (no Vec/array fields, whose `ListRepr` is
        // not yet ported): Outer.obj -> Boxed.typ -> TypeObj.base ->
        // Boxed, closing the loop with two back-edges. Each field is a
        // bare struct name → projects to `SomeInstance(inner classdef)`,
        // so setup recurses purely through `InstanceRepr`s.
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "Outer".to_string(),
            vec![
                ("len".to_string(), "usize".to_string()),
                ("obj".to_string(), "Boxed".to_string()),
            ],
        );
        reg.fields.insert(
            "Boxed".to_string(),
            vec![("typ".to_string(), "TypeObj".to_string())],
        );
        reg.fields.insert(
            "TypeObj".to_string(),
            vec![
                // cycle back-edge: TypeObj.base -> Boxed -> TypeObj
                ("base".to_string(), "Boxed".to_string()),
                // self-reference back-edge: TypeObj.parent -> TypeObj
                ("parent".to_string(), "TypeObj".to_string()),
            ],
        );
        ann.bookkeeper.set_struct_fields(std::rc::Rc::new(reg));
        let cd = ann
            .bookkeeper
            .getuniqueclassdef_for_struct_root("Outer")
            .expect("cyclic graph registers");

        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr (cyclic)");
        rtyper
            .call_all_setups()
            .expect("call_all_setups drains cyclic pending reprs");
        assert!(
            matches!(inst.lowleveltype(), LowLevelType::Ptr(_)),
            "cyclic instance repr lowered to a Ptr without overflow"
        );
    }

    #[test]
    fn instance_repr_create_instance_allocates_immortal_struct() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let host = HostObject::new_class("Bare", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let crate::annotator::description::DescEntry::Class(rc) = &entry else {
            unreachable!();
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).unwrap();
        Repr::setup(inst.as_ref() as &dyn Repr).unwrap();
        let p = inst.create_instance().expect("create_instance");
        assert!(p.nonzero());
    }

    #[test]
    fn instance_repr_convert_const_none_returns_null_instance_constant() {
        // rclass.py — `if value is None: return self.null_instance()`.
        // null_instance returns `nullptr(self.object_type)` wrapped as
        // ConstValue::LLPtr; concretetype tracks `self.lowleveltype`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("init exceptiondata");
        let host = HostObject::new_class("Bare2", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let DescEntry::Class(rc) = &entry else {
            unreachable!()
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        assert!(matches!(
            &inst.object_type,
            LowLevelType::ForwardReference(fwd) if fwd.resolved().is_none()
        ));

        let c = (inst.as_ref() as &dyn Repr)
            .convert_const(&ConstValue::None)
            .expect("convert_const(None)");
        assert_eq!(c.concretetype.as_ref(), Some(inst.lowleveltype()));
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("convert_const(None) must produce LLPtr, got {:?}", c.value);
        };
        assert!(!ptr.nonzero(), "null_instance must be a null pointer");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr after null creation");
        assert_eq!(c.concretetype.as_ref(), Some(inst.lowleveltype()));
        assert!(!ptr.nonzero());
    }

    #[test]
    fn instance_repr_root_convert_const_delegates_and_upcasts() {
        // rclass.py:786-790 delegates mismatched classdefs through the
        // concrete InstanceRepr and upcasts. For the root repr,
        // classdesc.py makes `classdef.commonbase(None) == None`,
        // so the guard is vacuously satisfied.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("init exceptiondata");
        let host = HostObject::new_class("RootDelegated", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let DescEntry::Class(rc) = &entry else {
            unreachable!()
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let concrete = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("concrete repr");
        Repr::setup(concrete.as_ref() as &dyn Repr).expect("setup concrete InstanceRepr");
        let root = getinstancerepr(&rtyper, None, Flavor::Gc).expect("root repr");
        assert!(root.classdef.is_none(), "test must exercise the root repr");
        Repr::setup(root.as_ref() as &dyn Repr).expect("setup root InstanceRepr");
        rtyper.call_all_setups().expect("call_all_setups");
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);

        let prebuilt = HostObject::new_instance(host.clone(), vec![]);
        let c = (root.as_ref() as &dyn Repr)
            .convert_const(&ConstValue::HostObject(prebuilt))
            .expect("root convert_const delegates");
        assert_eq!(c.concretetype.as_ref(), Some(root.lowleveltype()));
        let ConstValue::LLPtr(ptr) = &c.value else {
            panic!("convert_const must produce LLPtr, got {:?}", c.value);
        };
        assert!(ptr.nonzero());
        assert_eq!(
            concrete.iprebuiltinstances.borrow().len(),
            1,
            "root conversion must route through the concrete InstanceRepr"
        );
    }

    #[test]
    fn instance_repr_convert_const_exact_caches_prebuilt_instance() {
        // rclass.py — `convert_const_exact` populates
        // `iprebuiltinstances[value]` on first call. Re-issuing
        // `convert_const(value)` returns the same `_ptr` from the cache.
        // For a class with no instance attrs (`fields` empty), the
        // initialize_prebuilt_data branch only writes the root
        // `typeptr`, so the cache identity is observable through the
        // resulting `LLPtr` `_identity` equality.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::HostObject;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("init exceptiondata");
        let host = HostObject::new_class("PrebuiltCls", vec![]);
        let entry = ann.bookkeeper.getdesc(&host).unwrap();
        let DescEntry::Class(rc) = &entry else {
            unreachable!()
        };
        let cd = crate::annotator::classdesc::ClassDesc::getuniqueclassdef(rc).unwrap();
        let inst = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        // Drain pendingsetup so the associated ClassRepr (vtable_type
        // ForwardReference) is also resolved before
        // initialize_prebuilt_data writes typeptr.
        rtyper.call_all_setups().expect("call_all_setups");
        // ClassRepr.init_vtable wants `classdef.minid`, populated by
        // normalizecalls.assign_inheritance_ids.
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);

        // Build a prebuilt host instance so convert_const reaches the
        // exact-match arm.
        let prebuilt = HostObject::new_instance(host.clone(), vec![]);
        let value = ConstValue::HostObject(prebuilt.clone());
        let c1 = (inst.as_ref() as &dyn Repr)
            .convert_const(&value)
            .expect("convert_const(prebuilt)#1");
        let c2 = (inst.as_ref() as &dyn Repr)
            .convert_const(&value)
            .expect("convert_const(prebuilt)#2");
        let (ConstValue::LLPtr(p1), ConstValue::LLPtr(p2)) = (&c1.value, &c2.value) else {
            panic!(
                "convert_const must produce LLPtr, got {:?} / {:?}",
                c1.value, c2.value
            );
        };
        // upstream identity: `iprebuiltinstances[value]` returns the
        // same ptr each call. Pyre `_ptr` equality is by `_identity`
        // which derives from the underlying object identity.
        assert_eq!(
            p1._hashable_identity(),
            p2._hashable_identity(),
            "iprebuiltinstances cache must return the same _ptr identity"
        );
        // The cache holds exactly one entry for this prebuilt.
        assert_eq!(inst.iprebuiltinstances.borrow().len(), 1);
    }

    #[test]
    fn prebuilt_data_defers_nested_initialization_and_reads_descriptors() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::{SomeInteger, SomeValue};
        use crate::flowspace::model::HostObject;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper.initialize_exceptiondata().unwrap();
        let host_class = HostObject::new_class("PrebuiltDescriptor", vec![]);
        let cd = ann.bookkeeper.getuniqueclassdef(&host_class).unwrap();
        ClassDef::generalize_attr(
            &cd,
            "value",
            Some(SomeValue::Integer(SomeInteger::new(false, false))),
        )
        .unwrap();
        cd.borrow_mut()
            .attrs
            .get_mut("value")
            .unwrap()
            .modified(None)
            .unwrap();
        let repr = getinstancerepr(&rtyper, Some(&cd), Flavor::Gc).unwrap();
        Repr::setup(repr.as_ref()).unwrap();
        rtyper.call_all_setups().unwrap();
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);

        // Supply the live host descriptor after the field layout is known.
        // rclass.py:InstanceRepr.initialize_prebuilt_data must observe
        // getattr(value, name), even with a shadowing __dict__ entry.
        let reads = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let getter_reads = reads.clone();
        let getter = HostObject::new_native_callable(
            "PrebuiltDescriptor.value",
            Arc::new(move |_| {
                getter_reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(ConstValue::Int(42))
            }),
        );
        host_class.class_set(
            "value",
            ConstValue::HostObject(HostObject::new_property(
                "PrebuiltDescriptor.value",
                Some(getter),
                None,
                None,
            )),
        );
        let value = HostObject::new_instance(host_class, vec![]);
        value.instance_set("value", ConstValue::Int(7)).unwrap();
        let saved = Rc::new(RefCell::new(None));
        let saved_inner = saved.clone();
        let repr_inner = repr.clone();
        let reads_inner = reads.clone();
        INITIALIZE_DATA_FLATTENREC
            .with(|flat| {
                flat.call(Box::new(move || {
                    *saved_inner.borrow_mut() = Some(repr_inner.convert_const_exact(&value)?);
                    assert_eq!(
                        reads_inner.load(std::sync::atomic::Ordering::Relaxed),
                        0,
                        "nested prebuilt initialization must wait for the outer object"
                    );
                    Ok(())
                }))
            })
            .unwrap();
        assert_eq!(reads.load(std::sync::atomic::Ordering::Relaxed), 1);
        let constant = saved.borrow_mut().take().unwrap();
        let ConstValue::LLPtr(ptr) = constant.value else {
            panic!("typed instance")
        };
        let lltype::_ptr_obj::Struct(object) = ptr._obj().unwrap() else {
            panic!("struct")
        };
        let fields = repr.fields();
        let (field, _) = fields.get("value").expect("nonconstant instance field");
        assert!(matches!(
            object._getattr(field).unwrap(),
            lltype::LowLevelValue::Signed(42)
        ));
    }

    #[test]
    fn instance_repr_convert_const_non_hostobject_rejected() {
        // rclass.py (`bk.getuniqueclassdef(value.__class__)`) —
        // pyre's `getuniqueclassdef` only accepts HostObject inputs, so
        // the convert_const port rejects non-HostObject ConstValues
        // before reaching the bookkeeper.
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RejectShape", None);
        let inst = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        let err = (inst.as_ref() as &dyn Repr)
            .convert_const(&ConstValue::Int(7))
            .unwrap_err();
        assert!(
            err.to_string().contains("expected HostObject or None"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn instance_repr_convert_const_generic_unit_variant_uses_interned_variant_classdef() {
        // `InstanceRepr.convert_const` (`rclass.py`) does
        // `bk.getuniqueclassdef(value.__class__)`.  A folded unit-variant
        // instance of a generic enum used to carry a standalone class host
        // (`HostObject::new_class` of the dotted ctor path) whose ClassDef
        // was not the interned variant, so convert_const raised
        // `not an instance of "...::Continue"`.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::front::StructFieldRegistry;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let mut reg = StructFieldRegistry::default();
        reg.fields.insert(
            "pyopcode::StepResult".to_string(),
            vec![("__discriminant".to_string(), "i64".to_string())],
        );
        reg.fields.insert(
            "pyopcode::StepResult<*mut PyObject>::Continue".to_string(),
            vec![],
        );
        bk.set_struct_fields(Rc::new(reg));
        for root in bk.struct_root_names() {
            let _ = bk.getuniqueclassdef_for_struct_root(&root);
        }
        let variant_cd = bk
            .getuniqueclassdef_for_enum_variant(
                "pyre_interpreter::pyopcode::StepResult<*mut PyObject>",
                "Continue",
            )
            .expect("variant classdef");
        let interned_host = bk.intern_enum_variant_host(
            "pyre_interpreter::pyopcode::StepResult<*mut PyObject>",
            "Continue",
        );

        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let inst =
            getinstancerepr(&rtyper, Some(&variant_cd), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(inst.as_ref() as &dyn Repr).expect("setup InstanceRepr");
        rtyper.call_all_setups().expect("call_all_setups");
        crate::translator::rtyper::normalizecalls::assign_inheritance_ids(&ann);

        // Production intern attaches the prebuilt instance to the interned
        // variant host. convert_const then does
        // `bk.getuniqueclassdef(value.__class__)` (rclass.py) on that host.
        let linked =
            crate::translator::rtyper::unit_variant_fold::intern_unit_variant_prebuilt_instance(
                &bk,
                "pyre_interpreter.pyopcode.StepResult<*mut PyObject>.Continue",
                Some(0),
            )
            .expect("linked prebuilt");
        assert_eq!(
            linked.class_of().as_ref(),
            Some(&interned_host),
            "prebuilt instance class must be intern_enum_variant_host"
        );
        let converted = (inst.as_ref() as &dyn Repr)
            .convert_const(&ConstValue::HostObject(linked))
            .expect("convert_const of generic unit-variant prebuilt");
        assert!(
            matches!(converted.value, ConstValue::LLPtr(_)),
            "convert_const must produce LLPtr, got {:?}",
            converted.value
        );
    }

    #[test]
    fn classrepr_new_stores_classdef_and_unresolved_vtable_forward_reference() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = ClassRepr::new(&rtyper, &classdef);

        // rclass.py:194 — `self.classdef = classdef`.
        assert!(Rc::ptr_eq(&repr.classdef(), &classdef));

        // rclass.py:195 — vtable_type is a fresh ForwardReference (no
        // become() yet).
        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!("ClassRepr.vtable_type must be ForwardReference before setup");
        };
        assert!(
            fwd.resolved().is_none(),
            "vtable_type should not be resolved until _setup_repr runs"
        );

        // rclass.py:196 — `lowleveltype = Ptr(self.vtable_type)`.
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("ClassRepr.lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn classrepr_setup_resolves_vtable_struct_with_super_pointing_at_root_vtable() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr_arc =
            match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr_arc(classdef)") {
                ClassReprArc::Inst(r) => r,
                _ => panic!("classdef != None should return ClassReprArc::Inst"),
            };

        Repr::setup(repr_arc.as_ref()).expect("ClassRepr._setup_repr");

        // rclass.py:276-279 — vtable_type resolves to `<name>_vtable`
        // Struct with a single ('super', rbase.vtable_type) field.
        let LowLevelType::ForwardReference(fwd) = repr_arc.vtable_type() else {
            panic!("vtable_type must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("vtable_type must be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("vtable_type must resolve to Struct");
        };
        assert_eq!(body._name, "pkg.C_vtable");
        let super_field = body
            ._flds
            .get("super")
            .expect("vtable Struct must carry a 'super' field");

        // rbase for classdef with no basedef = RootClassRepr, whose
        // vtable_type is OBJECT_VTABLE (via rclass.py + rclass.py).
        assert_eq!(*super_field, OBJECT_VTABLE.clone());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("static_immutable"),
            Some(&ConstValue::Bool(true))
        );
    }

    #[test]
    fn getclassrepr_caches_second_lookup_returns_same_arc() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let first = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("first") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        let second = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("second") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        assert!(
            Arc::ptr_eq(&first, &second),
            "getclassrepr must cache by ClassDefKey identity"
        );
    }

    #[test]
    fn classrepr_basedef_recursion_threads_parent_vtable_type_into_super_field() {
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));

        // Force both reprs to exist so we can compare their vtable_types.
        let base_arc = match getclassrepr_arc(&rtyper, Some(&base)).expect("base") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        let child_arc = match getclassrepr_arc(&rtyper, Some(&child)).expect("child") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };

        // Setup child: internally triggers base.setup() via rclass.py:274.
        Repr::setup(child_arc.as_ref()).expect("child setup");

        // Both reprs must now be FINISHED.
        let resolved_base = match base_arc.vtable_type() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("base vtable_type must be resolved via child setup"),
            _ => panic!(),
        };
        let resolved_child = match child_arc.vtable_type() {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().expect("child vtable_type must be resolved")
            }
            _ => panic!(),
        };

        // Child's 'super' field type must equal base's vtable_type
        // (resolved LowLevelType::ForwardReference that compares equal
        // via inner Arc<Mutex> target).
        let LowLevelType::Struct(child_body) = resolved_child else {
            panic!("child vtable must be Struct");
        };
        let super_field = child_body
            ._flds
            .get("super")
            .expect("child vtable must have 'super'");
        // rclass.py passes `rbase.vtable_type` (a ForwardReference
        // LowLevelType wrapper around the same Arc<Mutex> that the
        // base repr holds). We stored a clone in the 'super' slot, so
        // both should resolve to the same inner Struct body.
        let LowLevelType::ForwardReference(super_fwd) = super_field else {
            panic!("'super' field type must be ForwardReference to base vtable");
        };
        assert_eq!(
            super_fwd
                .resolved()
                .expect("super field's ForwardReference resolved"),
            resolved_base,
        );
    }

    #[test]
    fn getclassrepr_arc_none_classdef_returns_rootclass_repr() {
        let rtyper = fresh_rtyper();
        let arc = getclassrepr_arc(&rtyper, None).expect("None classdef");
        match arc {
            ClassReprArc::Root(r) => assert_eq!(r.classdef(), None),
            _ => panic!("None classdef must route to Root"),
        }
    }

    #[test]
    fn getclassrepr_without_rootclass_repr_returns_typererror_for_none() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        // No initialize_exceptiondata — rootclass_repr stays None.
        let err =
            getclassrepr_arc(&rtyper, None).expect_err("should surface missing rootclass_repr");
        assert!(
            err.to_string().contains("initialize_exceptiondata"),
            "error must point at initialize_exceptiondata, got: {err}",
        );
    }

    #[test]
    fn classrepr_prepare_method_rewrites_method_pbc_to_function_pbc() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::description::{FunctionDesc, MethodDesc};
        use crate::flowspace::argument::Signature;

        let rtyper = fresh_rtyper();
        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.C".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        bk.register_classdef(classdef.clone());
        let funcdesc = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "pkg.C.method",
            Signature::new(vec!["self".to_string()], None, None),
            None,
            None,
        )));
        let method = Rc::new(RefCell::new(MethodDesc::new(
            bk,
            crate::annotator::description::FuncDescEntry::plain(funcdesc.clone()),
            ClassDefKey::from_classdef(&classdef),
            None,
            "method",
            BTreeMap::new(),
        )));
        let s_value = SomeValue::PBC(SomePBC::new(vec![DescEntry::Method(method)], true));

        let repr = ClassRepr::new(&rtyper, &classdef);
        let prepared = repr
            .prepare_method(&s_value)
            .expect("prepare_method")
            .expect("method SomePBC should be rewritten");
        let SomeValue::PBC(prepared_pbc) = prepared else {
            panic!("prepare_method must keep a SomePBC");
        };
        assert_eq!(
            prepared_pbc.get_kind().expect("PBC kind"),
            DescKind::Function
        );
        assert!(
            !prepared_pbc.can_be_none,
            "prepare_method must drop nullable method-PBC shape like upstream"
        );
        assert_eq!(prepared_pbc.descriptions.len(), 1);
        assert!(matches!(
            prepared_pbc
                .descriptions
                .values()
                .next()
                .expect("rewritten PBC entry"),
            crate::annotator::description::DescEntry::Func(_)
        ));
    }

    #[test]
    fn classrepr_setup_materializes_extra_access_sets() {
        use crate::annotator::description::ClassAttrFamily;

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let access_set = Rc::new(RefCell::new(ClassAttrFamily::new(
            classdef.borrow().classdesc.borrow().identity,
        )));
        access_set.borrow_mut().s_value = SomeValue::Bool(crate::annotator::model::SomeBool::new());
        let access_key = class_attr_family_key(&access_set);
        classdef
            .borrow_mut()
            .extra_access_sets
            .insert(access_key, (access_set, "x".to_string(), 0));

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("extra_access_sets setup");
        let pbcfields = repr.pbcfields.borrow();
        assert!(pbcfields.contains_key(&(access_key, "x".to_string())));
    }

    // R2-B — InstanceRepr classdef != None scaffolding.

    #[test]
    fn instance_repr_new_with_classdef_stores_forward_reference_object_type() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = InstanceRepr::new(&rtyper, Some(&classdef), Flavor::Gc);

        // rclass.py:470 — classdef field.
        let stored = repr.classdef().expect("classdef must be Some");
        assert!(Rc::ptr_eq(&stored, &classdef));

        // rclass.py:474-475 — object_type is a fresh GcForwardReference.
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference when classdef != None");
        };
        assert!(fwd.resolved().is_none());

        // rclass.py:477 — lowleveltype is Ptr(self.object_type).
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn buildinstancerepr_returns_virtualizable_instance_repr_when_class_declares_it() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::flowspace::model::{ConstValue, HostObject};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let mut members = indexmap::IndexMap::new();
        members.insert(
            "_virtualizable_".into(),
            ConstValue::List(vec![
                ConstValue::byte_str("x"),
                ConstValue::byte_str("items[*]"),
            ]),
        );
        let pyobj = HostObject::new_class_with_members("Frame", vec![], members);
        let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let classdesc = crate::annotator::classdesc::ClassDesc::new(
            &bk,
            pyobj,
            Some("Frame".into()),
            None,
            None,
        )
        .expect("ClassDesc");
        let classdef = crate::annotator::classdesc::ClassDef::new(&bk, &classdesc);
        let repr = buildinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("VirtualizableInstanceRepr");
        assert!(
            repr.virtualizable
                .borrow()
                .as_ref()
                .is_some_and(|vable| vable.top_of_virtualizable_hierarchy)
        );
    }

    #[test]
    fn virtualizable_setup_stamps_accessor_ranking_on_struct_hints() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::model::SomeInteger;
        use crate::flowspace::model::{ConstValue, HostObject};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let mut members = indexmap::IndexMap::new();
        members.insert(
            "_virtualizable_".into(),
            ConstValue::List(vec![ConstValue::byte_str("x")]),
        );
        let pyobj = HostObject::new_class_with_members("V", vec![], members);
        let bk = std::rc::Rc::new(crate::annotator::bookkeeper::Bookkeeper::new());
        let classdesc =
            crate::annotator::classdesc::ClassDesc::new(&bk, pyobj, Some("V".into()), None, None)
                .expect("ClassDesc");
        let classdef = crate::annotator::classdesc::ClassDef::new(&bk, &classdesc);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("vable repr");
        Repr::setup(repr.as_ref()).expect("VirtualizableInstanceRepr._setup_repr");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        assert!(
            body._flds.get("vable_token").is_some(),
            "vable_token precedes instance attrs"
        );
        let hint = body
            ._hints
            .get("virtualizable_accessor")
            .expect("VTYPE._hints['virtualizable_accessor']");
        let ConstValue::Opaque(opaque) = hint else {
            panic!("virtualizable_accessor hint must be the live accessor, got {hint:?}");
        };
        let accessor = opaque
            .downcast_ref::<parking_lot::Mutex<FieldListAccessor>>()
            .expect("hint is FieldListAccessor");
        let accessor = accessor.lock();
        assert_eq!(
            accessor.fields.get("inst_x").map(|rank| rank.name),
            Some("immutable"),
            "accessor.fields maps inst_x to IR_IMMUTABLE"
        );
        let accessor_ty = accessor.TYPE.clone().expect("accessor.TYPE");
        assert!(
            matches!(accessor_ty, LowLevelType::Struct(_)),
            "test_rvirtualizable.py test_accessor: accessor.TYPE == TYPE after become"
        );

        let child = ClassDef::new_standalone("W", Some(&classdef));
        let child_repr = getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("child");
        Repr::setup(child_repr.as_ref()).expect("subclass setup");
        let LowLevelType::ForwardReference(child_fwd) = child_repr.object_type() else {
            panic!("child object_type");
        };
        let child_resolved = child_fwd.resolved().expect("child resolved");
        let LowLevelType::Struct(child_body) = child_resolved else {
            panic!("child struct");
        };
        assert!(
            child_body._hints.get("virtualizable_accessor").is_none(),
            "test_rvirtualizable.py: subclass TYPE has no virtualizable_accessor"
        );
    }

    #[test]
    fn buildinstancerepr_unboxedvalue_without_taggedpointers_falls_back_to_plain_instance_repr() {
        use crate::annotator::annrpython::RPythonAnnotator;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let classdef = ClassDef::new_standalone("pkg.UnboxedValue", None);

        let repr =
            buildinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("plain InstanceRepr");

        let stored = repr.classdef().expect("classdef must be Some");
        assert!(Rc::ptr_eq(&stored, &classdef));
        Repr::setup(&repr).expect("plain InstanceRepr setup");
    }

    #[test]
    fn instance_repr_setup_resolves_object_type_to_struct_with_super_pointing_at_root_object() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("getinstancerepr(classdef)");

        Repr::setup(repr.as_ref()).expect("InstanceRepr._setup_repr");

        // rclass.py:548-554 — object_type resolves to Struct('pkg.C',
        // ('super', rbase.object_type)) where rbase is the root
        // InstanceRepr's OBJECT (classdef=None).
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must still be ForwardReference wrapper");
        };
        let resolved = fwd.resolved().expect("object_type must be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        assert_eq!(body._name, "pkg.C");
        assert_eq!(body._gckind, GcKind::Gc);
        let super_field = body
            ._flds
            .get("super")
            .expect("object_type Struct must carry 'super' field");
        assert_eq!(*super_field, OBJECT.clone());
    }

    #[test]
    fn instance_repr_gc_setup_attaches_runtime_type_info_opaque_on_body_struct() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RttiC", None);
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("getinstancerepr(classdef)");

        Repr::setup(repr.as_ref()).expect("InstanceRepr._setup_repr");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference wrapper");
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        // rclass.py:531-532 — gc-flavor instance reprs pass rtti=True
        // so `getRuntimeTypeInfo(body)` succeeds after setup.
        let rtti = body
            ._runtime_type_info
            .as_ref()
            .expect("gc-flavor instance body must carry _runtime_type_info");
        assert_eq!(rtti.TYPE.tag, "RuntimeTypeInfo");
        assert_eq!(rtti._name.as_deref(), Some("pkg.RttiC"));
    }

    #[test]
    fn instance_repr_raw_setup_does_not_attach_runtime_type_info_opaque() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RawC", None);
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::byte_str("raw"));
        let repr =
            getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr raw");
        assert_eq!(repr.gcflavor(), Flavor::Raw);
        Repr::setup(repr.as_ref()).expect("setup raw");
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("raw resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert!(body._runtime_type_info.is_none());
    }

    #[test]
    fn getinstancerepr_caches_by_classdef_identity() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let first = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("first");
        let second = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("second");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn getinstancerepr_reads_alloc_flavor_from_classdesc() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RawC", None);
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::byte_str("raw"));

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("raw repr");
        assert_eq!(repr.gcflavor(), Flavor::Raw);
    }

    #[test]
    fn instance_repr_basedef_recursion_threads_parent_object_type_into_super_field() {
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));
        let base_repr = getinstancerepr(&rtyper, Some(&base), Flavor::Gc).expect("base");
        let child_repr = getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("child");

        Repr::setup(child_repr.as_ref()).expect("child setup");

        // Both reprs must now be FINISHED.
        let LowLevelType::ForwardReference(child_fwd) = child_repr.object_type() else {
            panic!();
        };
        let child_resolved = child_fwd.resolved().expect("child resolved");
        let LowLevelType::ForwardReference(base_fwd) = base_repr.object_type() else {
            panic!();
        };
        let base_resolved = base_fwd.resolved().expect("base resolved via child");

        let LowLevelType::Struct(child_body) = child_resolved else {
            panic!();
        };
        let super_field = child_body
            ._flds
            .get("super")
            .expect("child object_type must have 'super'");
        let LowLevelType::ForwardReference(super_fwd) = super_field else {
            panic!("child 'super' field must be a ForwardReference pointing at base");
        };
        assert_eq!(
            super_fwd.resolved().expect("super field resolved"),
            base_resolved
        );
    }

    // R2-C — attrs iteration on ClassRepr + InstanceRepr.

    fn attach_attr(
        classdef: &Rc<RefCell<ClassDef>>,
        name: &str,
        s_value: SomeValue,
        readonly: bool,
    ) {
        use crate::annotator::classdesc::Attribute;
        let mut attr = Attribute::new(name);
        attr.s_value = s_value;
        attr.readonly = readonly;
        classdef.borrow_mut().attrs.insert(name.to_string(), attr);
    }

    #[test]
    fn classrepr_setup_populates_clsfields_for_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        let clsfields = repr.clsfields();
        let (mangled, field_repr) = clsfields
            .get("LIMIT")
            .expect("readonly attr LIMIT should land in clsfields");
        assert_eq!(mangled, "cls_LIMIT");
        // rclass.py:262 — llfields append uses r.lowleveltype; Integer's
        // Repr lowleveltype is lltype.Signed.
        assert_eq!(field_repr.lowleveltype(), &LowLevelType::Signed);

        // rclass.py:275-278 — the vtable Struct body must carry the
        // `(mangled_name, r.lowleveltype)` pair after 'super'.
        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("vtable_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        let limit_field = body
            ._flds
            .get("cls_LIMIT")
            .expect("vtable must carry cls_LIMIT field");
        assert_eq!(*limit_field, LowLevelType::Signed);
    }

    #[test]
    fn classrepr_setup_skips_non_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false, // instance attr, not class attr
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        assert!(
            repr.clsfields().is_empty(),
            "non-readonly attrs must not leak into clsfields"
        );
    }

    #[test]
    fn classrepr_sorts_vtable_fields_by_reverse_size_after_name_order() {
        use crate::annotator::model::{SomeBool, SomeInteger};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(&classdef, "a_bool", SomeValue::Bool(SomeBool::new()), true);
        attach_attr(
            &classdef,
            "z_int",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("vtable_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert_eq!(body._names[0], "super");
        assert_eq!(body._names[1], "cls_z_int");
        assert_eq!(body._names[2], "cls_a_bool");
    }

    #[test]
    fn instance_repr_setup_populates_fields_for_non_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(repr.as_ref()).expect("setup");

        let fields = repr.fields();
        let (mangled, field_repr) = fields.get("x").expect("instance attr x in fields");
        assert_eq!(mangled, "inst_x");
        assert_eq!(field_repr.lowleveltype(), &LowLevelType::Signed);

        // rclass.py:548-554 — object_type Struct body has inst_x pair.
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        let x_field = body
            ._flds
            .get("inst_x")
            .expect("object_type must carry inst_x field");
        assert_eq!(*x_field, LowLevelType::Signed);
    }

    #[test]
    fn instance_repr_allinstancefields_unions_parent_chain() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        attach_attr(
            &base,
            "shared",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));
        attach_attr(
            &child,
            "extra",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let child_repr =
            getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("getinstancerepr child");
        Repr::setup(child_repr.as_ref()).expect("child setup");

        let all = child_repr.allinstancefields();
        assert!(
            all.contains_key("shared"),
            "parent 'shared' attr must propagate into child.allinstancefields"
        );
        assert!(
            all.contains_key("extra"),
            "child's own 'extra' attr must appear in allinstancefields"
        );

        // Fields-proper stays minimal (only child's own).
        let fields = child_repr.fields();
        assert!(!fields.contains_key("shared"));
        assert!(fields.contains_key("extra"));
    }

    #[test]
    fn instance_repr_sorts_fields_by_reverse_size_after_name_order() {
        use crate::annotator::model::{SomeBool, SomeInteger};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(&classdef, "a_bool", SomeValue::Bool(SomeBool::new()), false);
        attach_attr(
            &classdef,
            "z_int",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(repr.as_ref()).expect("setup");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert_eq!(body._names[0], "super");
        assert_eq!(body._names[1], "inst_z_int");
        assert_eq!(body._names[2], "inst_a_bool");
    }

    // R3 — `ClassRepr::setup_vtable` + constant→lowlevel adapter.

    #[test]
    fn constant_to_lowlevel_value_unwraps_int_bool_float_ptr_and_void() {
        use crate::flowspace::model::Constant;
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };

        let signed = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        assert_eq!(
            constant_to_lowlevel_value(&signed).unwrap(),
            LowLevelValue::Signed(7)
        );

        // rint.convert_const(Bool) → Int(b as i64); adapter routes through Signed.
        let signed_from_bool =
            Constant::with_concretetype(ConstValue::Bool(true), LowLevelType::Signed);
        assert_eq!(
            constant_to_lowlevel_value(&signed_from_bool).unwrap(),
            LowLevelValue::Signed(1)
        );

        let bool_const = Constant::with_concretetype(ConstValue::Bool(false), LowLevelType::Bool);
        assert_eq!(
            constant_to_lowlevel_value(&bool_const).unwrap(),
            LowLevelValue::Bool(false)
        );

        let float_const = Constant::with_concretetype(
            ConstValue::Float(0x4020_0000_0000_0000),
            LowLevelType::Float,
        );
        assert_eq!(
            constant_to_lowlevel_value(&float_const).unwrap(),
            LowLevelValue::Float(0x4020_0000_0000_0000)
        );

        let void_const = Constant::with_concretetype(ConstValue::None, LowLevelType::Void);
        assert_eq!(
            constant_to_lowlevel_value(&void_const).unwrap(),
            LowLevelValue::Void
        );

        // LLPtr case — forge a _ptr via malloc and verify the adapter returns
        // a Ptr variant carrying the exact same _ptr identity.
        let struct_t = LowLevelType::Struct(Box::new(Struct::gc(
            "pkg.P",
            vec![("x".into(), LowLevelType::Signed)],
        )));
        let ptr = malloc(struct_t.clone(), None, MallocFlavor::Gc, true).unwrap();
        let ptr_t = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::Struct(Struct::gc(
                "pkg.P",
                vec![("x".into(), LowLevelType::Signed)],
            )),
        }));
        let ptr_const =
            Constant::with_concretetype(ConstValue::LLPtr(Box::new(ptr.clone())), ptr_t.clone());
        let llvalue = constant_to_lowlevel_value(&ptr_const).unwrap();
        match llvalue {
            LowLevelValue::Ptr(got) => assert_eq!(got._identity, ptr._identity),
            other => panic!("expected Ptr, got {other:?}"),
        }
    }

    #[test]
    fn constant_to_lowlevel_value_rejects_mismatched_shape() {
        use crate::flowspace::model::Constant;

        let bad = Constant::with_concretetype(ConstValue::Int(3), LowLevelType::Bool);
        let err = constant_to_lowlevel_value(&bad).unwrap_err().to_string();
        assert!(
            err.contains("unsupported ConstValue→LowLevelValue"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn setup_vtable_writes_readonly_int_attr_into_vtable_field() {
        use crate::annotator::classdesc::ClassDictEntry;
        use crate::annotator::model::SomeInteger;
        use crate::translator::rtyper::lltypesystem::lltype::{
            LowLevelValue, MallocFlavor, malloc,
        };

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        // Seed readonly class attr on ClassDef so _setup_repr populates
        // clsfields with ("cls_LIMIT", IntegerRepr).
        attach_attr(
            &classdef,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );
        // Seed the classdesc's dictionary so read_attribute("LIMIT")
        // returns Constant(Int(7)) — the concrete value upstream resolves
        // via self.classdef.classdesc.read_attribute(fldname, None).
        {
            let cdesc = classdef.borrow().classdesc.clone();
            cdesc
                .borrow_mut()
                .classdict
                .insert("LIMIT".into(), ClassDictEntry::constant(ConstValue::Int(7)));
        }

        let repr_arc = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(repr_arc.as_ref()).expect("_setup_repr");

        // Allocate the vtable container — malloc produces an immortal
        // gc-Struct _ptr whose type matches `repr.lowleveltype` after
        // ForwardReference resolution.
        let LowLevelType::ForwardReference(fwd) = repr_arc.vtable_type() else {
            panic!("vtable_type must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("resolved vtable_type");
        let mut vtable = malloc(resolved, None, MallocFlavor::Gc, true).expect("malloc vtable");

        repr_arc
            .setup_vtable(&mut vtable, &repr_arc)
            .expect("setup_vtable");

        // rclass.py — `setattr(vtable, mangled_name, llvalue)` is
        // observable via the struct's getattr.
        let value = vtable.getattr("cls_LIMIT").expect("cls_LIMIT stored");
        assert_eq!(value, LowLevelValue::Signed(7));
    }

    /// `init_vtable` walks the super-chain calling
    /// `self.setup_vtable(vtable_part, r_parentcls)` (rclass.py) —
    /// receiver is the leaf class, argument is the parent providing the
    /// field shape. The values therefore come from the leaf's classdesc
    /// (via `read_attribute` MRO walk), not from the parent's. A child
    /// that overrides a parent class attr must see the child's value at
    /// the parent's vtable slot.
    #[test]
    fn init_vtable_uses_leaf_classdesc_for_parent_level_field_values() {
        use crate::annotator::classdesc::ClassDictEntry;
        use crate::annotator::model::SomeInteger;
        use crate::translator::rtyper::lltypesystem::lltype::LowLevelValue;

        let rtyper = fresh_rtyper();

        // Parent P: declares readonly class attr `LIMIT` (so it lands in
        // P.clsfields after _setup_repr) with value 7 in classdict.
        let parent = ClassDef::new_standalone("pkg.P", None);
        attach_attr(
            &parent,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );
        {
            let cdesc = parent.borrow().classdesc.clone();
            cdesc
                .borrow_mut()
                .classdict
                .insert("LIMIT".into(), ClassDictEntry::constant(ConstValue::Int(7)));
        }

        // `fill_vtable_root` reads `self.classdef.{minid,maxid}` —
        // assign_inheritance_ids would normally populate these. Provide
        // distinct ranges for parent/child so the test does not depend
        // on that pass.
        {
            let mut p = parent.borrow_mut();
            p.minid = Some(2);
            p.maxid = Some(9);
        }

        // Child C extends P with override `LIMIT = 42` in C's classdict.
        // C does NOT redeclare the attribute so its own clsfields stay
        // empty — the only `cls_LIMIT` slot lives at the parent (super)
        // level of the vtable struct.
        let child = ClassDef::new_standalone("pkg.C", Some(&parent));
        {
            let mut c = child.borrow_mut();
            c.minid = Some(3);
            c.maxid = Some(8);
        }
        {
            let cdesc = child.borrow().classdesc.clone();
            cdesc.borrow_mut().classdict.insert(
                "LIMIT".into(),
                ClassDictEntry::constant(ConstValue::Int(42)),
            );
        }

        // Pre-setup the parent repr so its rbase + clsfields are
        // populated before C's setup walks the chain.
        let parent_repr = match getclassrepr_arc(&rtyper, Some(&parent)).expect("getclassrepr P") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(parent_repr.as_ref()).expect("_setup_repr P");

        let child_repr = match getclassrepr_arc(&rtyper, Some(&child)).expect("getclassrepr C") {
            ClassReprArc::Inst(r) => r,
            _ => panic!("classdef != None routes to Inst"),
        };
        Repr::setup(child_repr.as_ref()).expect("_setup_repr C");

        // Drive `init_vtable` on the leaf (child). Without the
        // leaf-receiver fix the parent-level slot would be filled from
        // P's classdesc (value 7) instead of C's override (42).
        child_repr.init_vtable().expect("init_vtable C");

        let raw = child_repr
            .vtable
            .borrow()
            .clone()
            .expect("init_vtable post-condition: vtable is Some");
        // C_vtable.super → P_vtable; P_vtable carries `cls_LIMIT`.
        // `_ptr.getattr` exposes substructs as a `Ptr` aliasing the
        // parent allocation, so chain `.getattr` calls drill down.
        let parent_slice = raw.getattr("super").expect("super substruct");
        let LowLevelValue::Ptr(parent_ptr) = parent_slice else {
            panic!("expected substruct ptr, got {parent_slice:?}");
        };
        let limit = parent_ptr.getattr("cls_LIMIT").expect("cls_LIMIT");
        assert_eq!(
            limit,
            LowLevelValue::Signed(42),
            "leaf-class override must win — value should come from C.classdesc"
        );
    }

    /// A Gc-opaque pointer converts to the classdef-less root instance with
    /// the GCREF edge's `cast_opaque_ptr`; a raw opaque pointer has no
    /// such edge.
    #[test]
    fn gc_opaque_ptr_converts_to_root_instance_by_cast_opaque_ptr() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, Ptr};
        use crate::translator::rtyper::rmodel::PtrRepr;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let r_root = getinstancerepr(&rtyper, None, Flavor::Gc).expect("root InstanceRepr");

        let gc_opaque = Ptr::from_container_type(LowLevelType::Opaque(Box::new(OpaqueType::gc(
            "MAJIT_REF_OPAQUE",
        ))))
        .unwrap();
        let r_ptr = PtrRepr::new(gc_opaque.clone());
        let v = Variable::new();
        v.set_concretetype(Some(LowLevelType::Ptr(Box::new(gc_opaque))));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let converted = super::super::pairtype::pair_convert_from_to(
            &r_ptr,
            r_root.as_ref(),
            &Hlvalue::Variable(v),
            &mut llops,
        )
        .expect("conversion does not error")
        .expect("Gc-opaque pointer converts to the root instance");
        assert_eq!(llops.ops.len(), 1);
        assert_eq!(llops.ops[0].opname, "cast_opaque_ptr");
        let Hlvalue::Variable(v_converted) = converted else {
            panic!("cast_opaque_ptr yields a Variable");
        };
        assert_eq!(
            v_converted.concretetype().as_ref(),
            Some(r_root.lowleveltype())
        );

        let raw_opaque =
            Ptr::from_container_type(LowLevelType::Opaque(Box::new(OpaqueType::new("RAW"))))
                .unwrap();
        let r_raw = PtrRepr::new(raw_opaque.clone());
        let v_raw = Variable::new();
        v_raw.set_concretetype(Some(LowLevelType::Ptr(Box::new(raw_opaque))));
        let mut llops = LowLevelOpList::new(rtyper.clone(), None);
        let converted = super::super::pairtype::pair_convert_from_to(
            &r_raw,
            r_root.as_ref(),
            &Hlvalue::Variable(v_raw),
            &mut llops,
        )
        .expect("conversion does not error");
        assert!(converted.is_none());
        assert!(llops.ops.is_empty());
    }
}
