//! RPython `rpython/rlib/rutf8.py` — UTF-8 helpers that the rtyper
//! synthesises as graphs rather than annotating the Python body.
//!
//! `class OutOfRange(Exception)` (`rutf8.py`) is a user rlib exception
//! whose `__init__` stores `self.code`. At the raise point in
//! `unichr_as_utf8` `code` is `r_uint`, so the annotator attribute is
//! `SomeInteger(unsigned=True)` (`model.py` `SomeInteger`).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::LazyLock;

use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::classdesc::ClassDef;
use crate::annotator::model::{AnnotatorError, SomeInteger, SomeValue};
use crate::flowspace::model::{BlockRef, ConstValue, HOST_ENV, Hlvalue, HostObject, Variable};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::rclass::{Flags, Flavor, getinstancerepr};
use crate::translator::rtyper::rmodel::Repr;
use crate::translator::rtyper::rtyper::{LowLevelOpList, RPythonTyper};

/// Process-global `rutf8.OutOfRange` class object (`rutf8.py`).
///
/// Qualname keeps a dot so `ClassDesc.is_builtin_exception_class` is
/// false (`classdesc.py`: no-dot names are the HOST_ENV builtins) and
/// `ClassDef.generalize_attr("code")` is allowed. Base is the HOST_ENV
/// `Exception` class (`HostObject::new_class`, the same constructor
/// `lookup_exception_class` documents for user classes).
static OUT_OF_RANGE: LazyLock<HostObject> = LazyLock::new(|| {
    let exception = HOST_ENV
        .lookup_exception_class("Exception")
        .expect("HOST_ENV Exception");
    HostObject::new_class("rutf8.OutOfRange", vec![exception])
});

/// `rutf8.OutOfRange` (`rutf8.py`).
pub(crate) fn out_of_range_class() -> HostObject {
    OUT_OF_RANGE.clone()
}

/// Intern `OutOfRange` on `bk` (`Bookkeeper.getuniqueclassdef`) and
/// record instance attribute `code` as unsigned (`OutOfRange.__init__`
/// `self.code = code` after `code = r_uint(code)`).
///
/// `Attribute.__init__` starts `readonly=True`; `Attribute.modified`
/// (`classdesc.py`) marks the `__init__` setattr so `InstanceRepr`
/// emits `inst_code` (`rclass.py` `_setup_repr`).
pub(crate) fn out_of_range_classdef(
    bk: &Rc<Bookkeeper>,
) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
    let classdef = bk.getuniqueclassdef(&out_of_range_class())?;
    ClassDef::generalize_attr(
        &classdef,
        "code",
        Some(SomeValue::Integer(SomeInteger::new(true, true))),
    )?;
    let owner = ClassDef::locate_attribute(&classdef, "code")?;
    {
        let mut owner_mut = owner.borrow_mut();
        let attr = owner_mut.attrs.get_mut("code").ok_or_else(|| {
            AnnotatorError::new("rutf8.OutOfRange: attribute 'code' missing after generalize_attr")
        })?;
        attr.modified(None)?;
    }
    Ok(classdef)
}

/// Look up the `OutOfRange` classdef that annotation interned
/// (`builtin_unichr_as_utf8` / `unichr_as_utf8` body).
/// `bookkeeper.getuniqueclassdef` is idempotent, so
/// `out_of_range_classdef` may run again; ids come only from the
/// single `perform_normalizations` / `assign_inheritance_ids` pass.
/// A classdef still unnumbered here is the late-subclass case
/// `assign_inheritance_ids` leaves unnumbered (`TooLateForNewSubclass`).
fn intern_out_of_range_classdef(
    rtyper: &RPythonTyper,
) -> Result<Rc<RefCell<ClassDef>>, TyperError> {
    let annotator = rtyper
        .annotator
        .upgrade()
        .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
    let classdef = out_of_range_classdef(&annotator.bookkeeper)
        .map_err(|err| TyperError::message(err.to_string()))?;
    let numbered = {
        let cd = classdef.borrow();
        cd.minid.is_some() && cd.maxid.is_some()
    };
    if !numbered {
        return Err(TyperError::message(
            "rutf8.OutOfRange classdef is unnumbered: annotating \
             unichr_as_utf8 (builtin_unichr_as_utf8 / out_of_range_classdef) \
             must intern it before perform_normalizations.assign_inheritance_ids",
        ));
    }
    Ok(classdef)
}

/// Emit `InstanceRepr.new_instance` (`malloc` + `setfield __class__`
/// from `ClassRepr.getvtable`) then `InstanceRepr.setfield` of `code`
/// (`OutOfRange.__init__` / `rclass.py`). Returns the except-link pair
/// `(etype, evalue)`.
///
/// `etype` is the CLASSTYPE vtable constant
/// `ExceptionData.r_exception_type.convert_const` /
/// `ClassRepr.convert_desc` produce for a class that has a ClassRepr.
pub(crate) fn raise_out_of_range_args(
    rtyper: &RPythonTyper,
    block: &BlockRef,
    code: &Variable,
) -> Result<Vec<Hlvalue>, TyperError> {
    let classdef = intern_out_of_range_classdef(rtyper)?;
    let rtyper_rc = rtyper.self_rc()?;
    let rinstance = getinstancerepr(&rtyper_rc, Some(&classdef), Flavor::Gc)?;
    Repr::setup(rinstance.as_ref() as &dyn Repr)?;
    let mut llops = LowLevelOpList::new(rtyper_rc.clone(), Some(block.clone()));
    let vptr = rinstance.new_instance(&mut llops, None, false)?;
    rinstance.setfield(
        vptr.clone(),
        "code",
        Hlvalue::Variable(code.clone()),
        &mut llops,
        false,
        &Flags::default(),
    )?;
    block.borrow_mut().operations.extend(llops.ops);
    let etype = rtyper
        .exceptiondata()?
        .r_exception_type
        .convert_const(&ConstValue::HostObject(out_of_range_class()))?;
    Ok(vec![Hlvalue::Constant(etype), vptr])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::Block;
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
    use crate::translator::rtyper::normalizecalls::assign_inheritance_ids;
    use crate::translator::rtyper::rclass::{CLASSTYPE, getclassrepr_arc, ll_issubclass};
    use crate::translator::rtyper::rtyper::variable_with_lltype;

    /// Annotation intern → one `assign_inheritance_ids` → exceptiondata,
    /// matching `RPythonAnnotator.simplify` / `perform_normalizations`
    /// then `RPythonTyper.specialize` / `exceptiondata.finish`.
    fn live_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        out_of_range_classdef(&ann.bookkeeper).expect("annotation intern OutOfRange");
        crate::annotator::exception::standard_exception_classdefs(&ann.bookkeeper)
            .expect("annotation intern standard exceptions");
        assign_inheritance_ids(&ann);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper.finish_exceptiondata().expect("finish_exceptiondata");
        (ann, rtyper)
    }

    fn classdef_vtable(
        rtyper: &Rc<RPythonTyper>,
        classdef: &Rc<RefCell<ClassDef>>,
    ) -> crate::translator::rtyper::lltypesystem::lltype::_ptr {
        let rclass = getclassrepr_arc(rtyper, Some(classdef)).expect("getclassrepr_arc");
        Repr::setup(&*rclass.as_repr()).expect("ClassRepr.setup");
        rclass.getruntime(&CLASSTYPE).expect("getruntime")
    }

    #[test]
    fn out_of_range_is_exception_subclass_not_valueerror() {
        let cls = out_of_range_class();
        let exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let value_error = HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("ValueError");
        assert!(cls.is_subclass_of(&exception));
        assert!(!cls.is_subclass_of(&value_error));
        assert_ne!(cls, value_error);
        assert!(cls.qualname().contains('.'));
    }

    #[test]
    fn out_of_range_classdef_records_unsigned_code() {
        let bk = Rc::new(Bookkeeper::new());
        let classdef = out_of_range_classdef(&bk).expect("out_of_range_classdef");
        let exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let exception_def = bk
            .getuniqueclassdef(&exception)
            .expect("Exception classdef");
        let value_error = HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("ValueError");
        let value_error_def = bk
            .getuniqueclassdef(&value_error)
            .expect("ValueError classdef");
        assert!(classdef.borrow().issubclass(&exception_def));
        assert!(!classdef.borrow().issubclass(&value_error_def));
        let cd = classdef.borrow();
        let attr = cd.attrs.get("code").expect("code attribute");
        assert!(!attr.readonly);
        match &attr.s_value {
            SomeValue::Integer(i) => {
                assert!(i.unsigned);
                assert!(i.nonneg);
            }
            other => panic!("code must be unsigned SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn out_of_range_instancerepr_has_inst_code_and_vtable_range() {
        let (ann, rtyper) = live_rtyper();
        let classdef = intern_out_of_range_classdef(&rtyper).expect("intern OutOfRange");
        let rinstance =
            getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(rinstance.as_ref() as &dyn Repr).expect("InstanceRepr.setup");
        let fields = rinstance.fields();
        let (mangled, field_repr) = fields.get("code").expect("code field");
        assert_eq!(mangled, "inst_code");
        assert!(matches!(field_repr.lowleveltype(), LowLevelType::Unsigned));

        let exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception");
        let exception_def = ann
            .bookkeeper
            .getuniqueclassdef(&exception)
            .expect("Exception classdef");
        let value_error = HOST_ENV
            .lookup_exception_class("ValueError")
            .expect("ValueError");
        let value_error_def = ann
            .bookkeeper
            .getuniqueclassdef(&value_error)
            .expect("ValueError classdef");
        let oor_vt = classdef_vtable(&rtyper, &classdef);
        let exc_vt = classdef_vtable(&rtyper, &exception_def);
        let ve_vt = classdef_vtable(&rtyper, &value_error_def);
        assert!(oor_vt.nonzero());
        assert!(ll_issubclass(&oor_vt, &exc_vt).expect("ll_issubclass Exception"));
        assert!(!ll_issubclass(&oor_vt, &ve_vt).expect("ll_issubclass ValueError"));
    }

    #[test]
    fn raise_out_of_range_args_sets_typeptr_and_inst_code() {
        let (_ann, rtyper) = live_rtyper();
        let classdef = intern_out_of_range_classdef(&rtyper).expect("intern OutOfRange");
        let oor_vt = classdef_vtable(&rtyper, &classdef);
        let code = variable_with_lltype("code", LowLevelType::Unsigned);
        let block = Block::shared(vec![Hlvalue::Variable(code.clone())]);
        let args =
            raise_out_of_range_args(&rtyper, &block, &code).expect("raise_out_of_range_args");
        let ops = block.borrow().operations.clone();
        assert!(
            ops.iter().any(|op| op.opname == "malloc"),
            "new_instance must malloc"
        );
        let typeptr_set = ops
            .iter()
            .find(|op| {
                op.opname == "setfield"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c))
                            if matches!(&c.value, ConstValue::ByteStr(s) if s == b"typeptr")
                    )
            })
            .expect("setfield typeptr");
        match typeptr_set.args.get(2) {
            Some(Hlvalue::Constant(c)) => match &c.value {
                ConstValue::LLPtr(p) => {
                    assert!(p.nonzero());
                    assert_eq!(p.as_ref(), &oor_vt);
                    assert_eq!(c.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
                }
                other => panic!("typeptr must be OutOfRange vtable, got {other:?}"),
            },
            other => panic!("typeptr value must be a Constant, got {other:?}"),
        }
        let set_code = ops
            .iter()
            .find(|op| {
                op.opname == "setfield"
                    && matches!(
                        op.args.get(1),
                        Some(Hlvalue::Constant(c))
                            if matches!(&c.value, ConstValue::ByteStr(s) if s == b"inst_code")
                    )
            })
            .expect("setfield inst_code");
        match set_code.args.get(2) {
            Some(Hlvalue::Variable(v)) => assert_eq!(v, &code),
            other => panic!("inst_code must be the input code, got {other:?}"),
        }
        assert_eq!(args.len(), 2);
        match &args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::LLPtr(p) => {
                    assert!(p.nonzero());
                    assert_eq!(p.as_ref(), &oor_vt);
                    assert_eq!(c.concretetype.as_ref(), Some(&CLASSTYPE.clone()));
                }
                other => panic!("etype must be OutOfRange vtable, got {other:?}"),
            },
            other => panic!("etype must be a Constant, got {other:?}"),
        }
        match &args[1] {
            Hlvalue::Variable(_) => {}
            other => panic!("evalue must be the malloc'd instance Variable, got {other:?}"),
        }
    }
}
