use majit_ir::{GcRef, GreenKey, OpRef, Type, Value};
use majit_macros::jit_driver;
use majit_metainterp::{
    DeclarativeJitDriver, JitDriver, JitState, PendingFieldWriteLayout, TraceCtx,
    recorder::{Snapshot, SnapshotFrame, SnapshotTagged},
    resume::{MaterializedValue, MaterializedVirtual, ResumeDataExt, ResumeDataVirtualAdder},
    virtualizable::VirtualizableInfo,
};

/// resoperation.py/727/739 `inputarg_from_tp(tp)` parity: mint a
/// sequence of typed `InputArg{Int,Float,Ref}` OpRefs for a test's
/// `JitState::create_sym`. The caller passes the same `Type` slice
/// that `extract_live_values` produces values for, so the sym entries
/// match the production typed inputargs the tracer would record.
#[allow(dead_code)]
fn typed_inputarg_sym(types: &[Type]) -> Vec<OpRef> {
    types
        .iter()
        .enumerate()
        .map(|(i, &tp)| OpRef::input_arg_typed(i as u32, tp))
        .collect()
}

#[allow(dead_code)]
fn attach_single_frame_snapshot(ctx: &mut TraceCtx, pc: u32, boxes: &[(OpRef, Type)]) {
    let snapshot_id = ctx.capture_resumedata(Snapshot {
        resume_position: -1,
        frames: vec![SnapshotFrame {
            jitcode_index: 0,
            pc,
            boxes: boxes
                .iter()
                .map(|(opref, tp)| SnapshotTagged::Box(*opref, *tp))
                .collect(),
        }],
        vable_boxes: Vec::new(),
        vref_boxes: Vec::new(),
    });
    ctx.set_last_guard_resume_position(snapshot_id);
}

#[jit_driver(greens = [pc, code], reds = [frame, stack], virtualizable = frame)]
struct DeclarativeDriver;

#[jit_driver(greens = [pc, code], reds = [frame, acc], virtualizable = frame)]
struct TypedDeclarativeDriver;

#[jit_driver(greens = [pc, code], reds = [frame, stackpos, top], virtualizable = frame)]
struct AutoVirtualizableDriver;

#[derive(Clone, Debug, PartialEq, Eq)]
struct TestMeta {
    header_pc: usize,
}

struct TestState {
    frame: usize,
    stack: i64,
}

impl JitState for TestState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.stack)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stack = values[1].as_int();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct BadLiveState {
    frame: usize,
}

impl JitState for BadLiveState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame))]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct ExtraLiveState {
    frame: usize,
    stack: i64,
    extra: i64,
}

impl JitState for ExtraLiveState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack, self.extra]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.frame)),
            Value::Int(self.stack),
            Value::Int(self.extra),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
        self.extra = values[2];
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct TypedState {
    frame: usize,
    acc: f64,
}

impl JitState for TypedState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.acc.to_bits() as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Float(self.acc)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Float])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.acc = f64::from_bits(values[1] as u64);
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.acc = values[1].as_float();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct MismatchedTypedState {
    frame: i64,
    acc: f64,
}

#[expect(dead_code, reason = "fixture type used by trait impl parity coverage")]
struct TypedRestoreOnlyState {
    frame: usize,
    acc: f64,
    raw_restore_calls: usize,
    typed_restore_calls: usize,
}

impl JitState for MismatchedTypedState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame, self.acc.to_bits() as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Int(self.frame), Value::Float(self.acc)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Int, Type::Float])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0];
        self.acc = f64::from_bits(values[1] as u64);
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for TypedRestoreOnlyState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.acc.to_bits() as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Float(self.acc)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Float])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
        self.raw_restore_calls += 1;
        self.frame = 0;
        self.acc = -1.0;
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.typed_restore_calls += 1;
        self.frame = values[0].as_ref().as_usize();
        self.acc = values[1].as_float();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Float)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

struct VirtualizableSyncState {
    frame: usize,
    stack: i64,
    before_sync_calls: usize,
    after_sync_calls: usize,
    reject_sync: bool,
}

impl JitState for VirtualizableSyncState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.stack)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stack = values[1].as_int();
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> bool {
        assert_eq!(virtualizable, "frame");
        self.before_sync_calls += 1;
        !self.reject_sync
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) {
        assert_eq!(virtualizable, "frame");
        self.after_sync_calls += 1;
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

#[expect(dead_code, reason = "fixture type used by trait impl parity coverage")]
struct NamedVirtualizableSyncState {
    frame: usize,
    stack: i64,
    before_sync_calls: usize,
    after_sync_calls: usize,
    saw_info_on_before: bool,
    saw_info_on_after: bool,
}

#[repr(C)]
#[expect(
    dead_code,
    reason = "heap-layout fixture for auto-virtualizable parity"
)]
struct AutoVirtualizableFrame {
    token: u64,
    stackpos: i64,
    stack_ptr: *mut u8,
}

#[expect(dead_code, reason = "fixture type used by trait impl parity coverage")]
struct AutoVirtualizableState {
    frame: usize,
    stackpos: i64,
    top: i64,
}

struct ResumeMappedState {
    frame: usize,
    stackpos: i64,
    top: i64,
}

struct VirtualResumeState {
    obj: usize,
    flag: i64,
    materialized_ref: usize,
    materialize_calls: usize,
    /// `history.py:125 id(descr)` parity — kept here so the test asserts
    /// Arc identity rather than numeric `index()` (latter still passes
    /// for any unrelated descr that happens to share `index()`).
    expected_descr: Option<majit_ir::DescrRef>,
}

#[repr(C)]
struct PendingWriteCell {
    field: i64,
}

struct PendingWriteState {
    obj: usize,
    flag: i64,
}

struct PendingArrayWriteState {
    array: usize,
    flag: i64,
}

#[repr(C)]
struct PendingRefCell {
    child: usize,
}

struct PendingVirtualWriteState {
    parent: usize,
    child: usize,
    parent_ref: usize,
    child_ref: usize,
    materialize_order: Vec<usize>,
    /// `history.py:125 id(descr)` parity — Arc identity is the contract;
    /// numeric `index()` would still pass if reconstruction fabricates a
    /// different descr with the same index.
    expected_parent_descr: Option<majit_ir::DescrRef>,
    expected_child_descr: Option<majit_ir::DescrRef>,
}

struct MultiFrameResumeState {
    frame: usize,
    flag: i64,
    restored_pcs: Vec<u64>,
}

struct GenericMultiFrameResumeState {
    frame: usize,
    flag: i64,
    restored_pcs: Vec<u64>,
    restored_frames: Vec<(usize, i64)>,
    materialized_ref: usize,
    materialize_calls: usize,
}

struct NestedVirtualResumeState {
    outer: usize,
    inner: usize,
    alias: usize,
    outer_ref: usize,
    inner_ref: usize,
    materialize_order: Vec<usize>,
    /// `history.py:125 id(descr)` parity — keep the test sensitive to
    /// descr Arc identity, not just numeric `index()`.
    expected_inner_descr: Option<majit_ir::DescrRef>,
    expected_outer_descr: Option<majit_ir::DescrRef>,
}

impl JitState for NamedVirtualizableSyncState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stack]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.stack)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stack = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stack = values[1].as_int();
    }

    fn sync_named_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) -> bool {
        assert_eq!(virtualizable, "frame");
        self.before_sync_calls += 1;
        self.saw_info_on_before = info.is_some();
        true
    }

    fn sync_named_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) {
        assert_eq!(virtualizable, "frame");
        self.after_sync_calls += 1;
        self.saw_info_on_after = info.is_some();
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for AutoVirtualizableState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stackpos, self.top]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.frame)),
            Value::Int(self.stackpos),
            Value::Int(self.top),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stackpos = values[1];
        self.top = values[2];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stackpos = values[1].as_int();
        self.top = values[2].as_int();
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        assert_eq!(virtualizable, "frame");
        Some(self.frame as *mut u8)
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        assert_eq!(virtualizable, "frame");
        Some(vec![1])
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
        boxes: &[i64],
        array_lengths: &[usize],
    ) -> bool {
        assert_eq!(virtualizable, "frame");
        assert_eq!(array_lengths, &[1]);
        self.stackpos = boxes[0];
        self.top = boxes[1];
        true
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<usize>)> {
        assert_eq!(virtualizable, "frame");
        Some((vec![self.stackpos, self.top], vec![1]))
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![
            (sym[0], Type::Ref),
            (sym[1], Type::Int),
            (sym[2], Type::Int),
        ]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for ResumeMappedState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.stackpos, self.top]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.frame)),
            Value::Int(self.stackpos),
            Value::Int(self.top),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.stackpos = values[1];
        self.top = values[2];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.stackpos = values[1].as_int();
        self.top = values[2].as_int();
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        vec![Type::Ref, Type::Int, Type::Int]
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![
            (sym[0], Type::Ref),
            (sym[1], Type::Int),
            (sym[2], Type::Int),
        ]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for VirtualResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.obj as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.obj)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.obj = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.obj = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        vec![Type::Ref, Type::Int]
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        assert_eq!(virtual_index, 0);
        match materialized {
            MaterializedVirtual::Struct { descr, .. } => {
                let expected = self.expected_descr.as_ref().expect("set in test");
                assert!(matches!(
                    descr.as_ref(),
                    Some(actual) if std::sync::Arc::ptr_eq(actual, expected)
                ));
            }
            other => panic!("unexpected virtual materialization: {other:?}"),
        }
        self.materialize_calls += 1;
        Some(GcRef(self.materialized_ref))
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for PendingWriteState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.obj as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.obj)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.obj = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.obj = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        vec![Type::Ref, Type::Int]
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr: Option<&majit_ir::DescrRef>,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        let fd = descr?.as_field_descr()?;
        Some(PendingFieldWriteLayout::Field {
            offset: fd.offset(),
            value_type: fd.field_type(),
        })
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for PendingArrayWriteState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.array as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.array)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.array = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.array = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn live_value_types(&self, _meta: &Self::Meta) -> Vec<Type> {
        vec![Type::Ref, Type::Int]
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr: Option<&majit_ir::DescrRef>,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        let ad = descr?.as_array_descr()?;
        Some(PendingFieldWriteLayout::ArrayItem {
            base_offset: ad.base_size(),
            item_size: ad.item_size(),
            item_type: ad.item_type(),
        })
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for MultiFrameResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn restore_reconstructed_frames(
        &mut self,
        _meta: &Self::Meta,
        reconstructed_state: &majit_metainterp::resume::ReconstructedState,
        _materialized_virtuals: &[MaterializedVirtual],
        _exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        if reconstructed_state.frames.len() != 2 {
            return false;
        }
        self.restored_pcs = reconstructed_state
            .frames
            .iter()
            .map(|frame| frame.pc)
            .collect();
        let inner = reconstructed_state
            .frames
            .last()
            .expect("two frames required");
        self.frame = inner.values[0].lossy_i64() as usize;
        self.flag = inner.values[1].lossy_i64();
        true
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for GenericMultiFrameResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.frame as i64, self.flag]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![Value::Ref(GcRef(self.frame)), Value::Int(self.flag)]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Int])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.frame = values[0] as usize;
        self.flag = values[1];
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.frame = values[0].as_ref().as_usize();
        self.flag = values[1].as_int();
    }

    fn reconstructed_frame_value_types(
        &self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        Some(vec![Type::Ref, Type::Int])
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame_pc: u64,
        values: &[Value],
        _exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        self.restored_pcs.push(frame_pc);
        self.restored_frames
            .push((values[0].as_ref().as_usize(), values[1].as_int()));
        if frame_index + 1 == total_frames {
            self.frame = values[0].as_ref().as_usize();
            self.flag = values[1].as_int();
        }
        true
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        _virtual: &MaterializedVirtual,
    ) -> Option<GcRef> {
        assert_eq!(virtual_index, 0);
        self.materialize_calls += 1;
        Some(GcRef(self.materialized_ref))
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr: Option<&majit_ir::DescrRef>,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        let fd = descr?.as_field_descr()?;
        Some(PendingFieldWriteLayout::Field {
            offset: fd.offset(),
            value_type: fd.field_type(),
        })
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Int)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for PendingVirtualWriteState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.parent as i64, self.child as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.parent)),
            Value::Ref(GcRef(self.child)),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Ref])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.parent = values[0] as usize;
        self.child = values[1] as usize;
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.parent = values[0].as_ref().as_usize();
        self.child = values[1].as_ref().as_usize();
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        _materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_order.push(virtual_index);
        match (virtual_index, materialized) {
            (0, MaterializedVirtual::Struct { descr, .. }) => {
                let expected = self.expected_parent_descr.as_ref().expect("set in test");
                assert!(matches!(
                    descr.as_ref(),
                    Some(actual) if std::sync::Arc::ptr_eq(actual, expected)
                ));
                Some(GcRef(self.parent_ref))
            }
            (1, MaterializedVirtual::Struct { descr, .. }) => {
                let expected = self.expected_child_descr.as_ref().expect("set in test");
                assert!(matches!(
                    descr.as_ref(),
                    Some(actual) if std::sync::Arc::ptr_eq(actual, expected)
                ));
                Some(GcRef(self.child_ref))
            }
            other => panic!("unexpected pending-write virtual materialization: {other:?}"),
        }
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        descr: Option<&majit_ir::DescrRef>,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        let fd = descr?.as_field_descr()?;
        Some(PendingFieldWriteLayout::Field {
            offset: fd.offset(),
            value_type: fd.field_type(),
        })
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![(sym[0], Type::Ref), (sym[1], Type::Ref)]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

impl JitState for NestedVirtualResumeState {
    type Meta = TestMeta;
    type Sym = Vec<OpRef>;
    type Env = ();

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        TestMeta { header_pc }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        vec![self.outer as i64, self.inner as i64, self.alias as i64]
    }

    fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
        vec![
            Value::Ref(GcRef(self.outer)),
            Value::Ref(GcRef(self.inner)),
            Value::Ref(GcRef(self.alias)),
        ]
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        typed_inputarg_sym(&[Type::Ref, Type::Ref, Type::Ref])
    }

    fn is_compatible(&self, _meta: &Self::Meta) -> bool {
        true
    }

    fn restore(&mut self, _meta: &Self::Meta, values: &[i64]) {
        self.outer = values[0] as usize;
        self.inner = values[1] as usize;
        self.alias = values[2] as usize;
    }

    fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
        self.outer = values[0].as_ref().as_usize();
        self.inner = values[1].as_ref().as_usize();
        self.alias = values[2].as_ref().as_usize();
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_order.push(virtual_index);
        match (virtual_index, materialized) {
            (0, MaterializedVirtual::Struct { descr, fields, .. }) => {
                let expected = self.expected_inner_descr.as_ref().expect("set in test");
                assert!(matches!(
                    descr.as_ref(),
                    Some(actual) if std::sync::Arc::ptr_eq(actual, expected)
                ));
                assert_eq!(fields, &vec![(0, MaterializedValue::Value(77))]);
                Some(GcRef(self.inner_ref))
            }
            (1, MaterializedVirtual::Obj { descr, fields, .. }) => {
                let expected = self.expected_outer_descr.as_ref().expect("set in test");
                assert!(matches!(
                    descr.as_ref(),
                    Some(actual) if std::sync::Arc::ptr_eq(actual, expected)
                ));
                assert_eq!(fields[0], (0, MaterializedValue::VirtualRef(0)));
                assert_eq!(fields[1], (1, MaterializedValue::Value(99)));
                assert_eq!(
                    materialized_refs.first().copied().flatten(),
                    Some(GcRef(self.inner_ref))
                );
                let resolved = materialized
                    .resolve_with_refs(materialized_refs)
                    .expect("nested refs should resolve");
                match resolved {
                    MaterializedVirtual::Obj { fields, .. } => {
                        assert_eq!(
                            fields[0],
                            (0, MaterializedValue::Value(self.inner_ref as i64))
                        );
                        assert_eq!(fields[1], (1, MaterializedValue::Value(99)));
                    }
                    other => panic!("unexpected resolved virtual: {other:?}"),
                }
                Some(GcRef(self.outer_ref))
            }
            other => panic!("unexpected materialization: {other:?}"),
        }
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.clone()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        vec![
            (sym[0], Type::Ref),
            (sym[1], Type::Ref),
            (sym[2], Type::Ref),
        ]
    }

    fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
        true
    }
}

#[test]
fn runtime_driver_preserves_structured_green_key_and_descriptor_on_trace_start() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let green_key = DeclarativeDriver::green_key(&[7, 99]).expect("green key should build");
    let mut driver = JitDriver::<TestState>::with_descriptor(2, descriptor.clone());
    let mut state = TestState {
        frame: 10,
        stack: 20,
    };

    assert!(
        driver
            .back_edge_structured(
                green_key.hash_u64(),
                || green_key.clone(),
                7,
                &mut state,
                &(),
                || {}
            )
            .is_none()
    );
    // Crossing the threshold starts tracing, which returns the target pc so the
    // caller re-enters the dispatch loop where the trace heads. `None` would
    // mean no trace started at all.
    assert_eq!(
        driver
            .back_edge_structured(
                green_key.hash_u64(),
                || green_key.clone(),
                7,
                &mut state,
                &(),
                || {}
            )
            .and_then(|resume| resume.resume_pc()),
        Some(7)
    );
    assert!(driver.is_tracing());

    let ctx = driver
        .meta_interp_mut()
        .trace_ctx()
        .expect("trace must be active");
    assert_eq!(ctx.green_key(), green_key.hash_u64());
    assert_eq!(ctx.green_key_values(), Some(&green_key));
    let trace_descriptor = ctx
        .driver_descriptor()
        .expect("descriptor must be attached to trace");
    assert_eq!(trace_descriptor.virtualizable.as_deref(), Some("frame"));
    assert_eq!(trace_descriptor.vars, descriptor.vars);
}

#[test]
fn runtime_driver_attaches_descriptor_on_keyed_trace_start_without_structured_green_key() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let key = GreenKey::new(vec![11, 22]).hash_u64();
    let mut driver = JitDriver::<TestState>::with_descriptor(2, descriptor.clone());
    let mut state = TestState { frame: 3, stack: 4 };

    assert!(
        driver
            .back_edge_keyed(key, 11, &mut state, &(), || {})
            .is_none()
    );
    // Crossing the threshold starts tracing, which returns the target pc so the
    // caller re-enters the dispatch loop where the trace heads. `None` would
    // mean no trace started at all.
    assert_eq!(
        driver
            .back_edge_keyed(key, 11, &mut state, &(), || {})
            .and_then(|resume| resume.resume_pc()),
        Some(11)
    );
    assert!(driver.is_tracing());

    let ctx = driver
        .meta_interp_mut()
        .trace_ctx()
        .expect("trace must be active");
    assert_eq!(ctx.green_key(), key);
    assert_eq!(ctx.green_key_values(), None);
    let trace_descriptor = ctx
        .driver_descriptor()
        .expect("descriptor must be attached to trace");
    assert_eq!(
        trace_descriptor
            .virtualizable()
            .map(|var| var.name.as_str()),
        Some("frame")
    );
    assert_eq!(trace_descriptor.vars, descriptor.vars);
}

#[test]
fn declarative_driver_trait_builds_runtime_driver_without_manual_descriptor_plumbing() {
    let mut driver = JitDriver::<TestState>::with_declarative::<DeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int],
    )
    .expect("declarative driver should build runtime descriptor");
    let mut state = TestState { frame: 1, stack: 2 };

    assert!(
        driver
            .back_edge_declarative::<DeclarativeDriver>(&[13, 21], 13, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    // Crossing the threshold starts tracing, which returns the target pc so the
    // caller re-enters the dispatch loop where the trace heads. `None` would
    // mean no trace started at all.
    assert_eq!(
        driver
            .back_edge_declarative::<DeclarativeDriver>(&[13, 21], 13, &mut state, &(), || {})
            .expect("green key should build"),
        Some(13)
    );
    assert!(driver.is_tracing());

    let ctx = driver
        .meta_interp_mut()
        .trace_ctx()
        .expect("trace must be active");
    assert_eq!(ctx.green_key_values(), Some(&GreenKey::new(vec![13, 21])));
    assert_eq!(
        ctx.driver_descriptor()
            .and_then(|descr| descr.virtualizable().map(|var| var.name.as_str())),
        Some("frame")
    );
}

#[test]
fn declarative_driver_trait_matches_inherent_green_key_builder() {
    let via_trait = <DeclarativeDriver as DeclarativeJitDriver>::green_key(&[5, 8])
        .expect("trait green key should build");
    let via_inherent =
        DeclarativeDriver::green_key(&[5, 8]).expect("inherent green key should build");
    assert_eq!(via_trait, via_inherent);
}

#[test]
fn declarative_driver_rejects_live_value_count_mismatch() {
    let mut driver = JitDriver::<BadLiveState>::with_declarative::<DeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int],
    )
    .expect("declarative driver should build runtime descriptor");
    let mut state = BadLiveState { frame: 9 };

    assert!(
        driver
            .back_edge_declarative::<DeclarativeDriver>(&[1, 2], 1, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    assert!(
        driver
            .back_edge_declarative::<DeclarativeDriver>(&[1, 2], 1, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    assert!(!driver.is_tracing());
}

#[test]
fn declarative_driver_rejects_extra_live_values_beyond_descriptor_reds() {
    let mut driver = JitDriver::<ExtraLiveState>::with_declarative::<DeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Int],
    )
    .expect("declarative driver should build runtime descriptor");
    let mut state = ExtraLiveState {
        frame: 9,
        stack: 4,
        extra: 17,
    };

    assert!(
        driver
            .back_edge_declarative::<DeclarativeDriver>(&[1, 2], 1, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    assert!(
        driver
            .back_edge_declarative::<DeclarativeDriver>(&[1, 2], 1, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    assert!(!driver.is_tracing());
}

#[test]
fn declarative_driver_preserves_typed_red_inputargs_on_trace_start() {
    let mut driver = JitDriver::<TypedState>::with_declarative::<TypedDeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Float],
    )
    .expect("typed declarative driver should build runtime descriptor");
    let mut state = TypedState {
        frame: 0x1234,
        acc: 3.5,
    };

    assert!(
        driver
            .back_edge_declarative::<TypedDeclarativeDriver>(&[17, 23], 17, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    // Crossing the threshold starts tracing, which returns the target pc so the
    // caller re-enters the dispatch loop where the trace heads. `None` would
    // mean no trace started at all.
    assert_eq!(
        driver
            .back_edge_declarative::<TypedDeclarativeDriver>(&[17, 23], 17, &mut state, &(), || {})
            .expect("green key should build"),
        Some(17)
    );
    assert!(driver.is_tracing());

    let (trace, _) = driver
        .meta_interp_mut()
        .finish_trace_for_parity(&typed_inputarg_sym(&[Type::Ref, Type::Float]))
        .expect("trace should be finishable for parity");
    assert_eq!(trace.inputargs.len(), 2);
    assert_eq!(trace.inputargs[0].tp.get(), Type::Ref);
    assert_eq!(trace.inputargs[1].tp.get(), Type::Float);
}

#[test]
fn declarative_driver_rejects_live_value_type_mismatch() {
    let mut driver = JitDriver::<MismatchedTypedState>::with_declarative::<TypedDeclarativeDriver>(
        2,
        &[Type::Int, Type::Int],
        &[Type::Ref, Type::Float],
    )
    .expect("typed declarative driver should build runtime descriptor");
    let mut state = MismatchedTypedState {
        frame: 7,
        acc: 1.25,
    };

    assert!(
        driver
            .back_edge_declarative::<TypedDeclarativeDriver>(&[19, 29], 19, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    assert!(
        driver
            .back_edge_declarative::<TypedDeclarativeDriver>(&[19, 29], 19, &mut state, &(), || {})
            .expect("green key should build")
            .is_none()
    );
    assert!(!driver.is_tracing());
}

#[test]
fn declarative_driver_can_reject_virtualizable_sync_after_the_counter_fires() {
    let descriptor =
        DeclarativeDriver::descriptor(&[Type::Int, Type::Int], &[Type::Ref, Type::Int])
            .expect("descriptor should build");
    let mut driver = JitDriver::<VirtualizableSyncState>::with_descriptor(2, descriptor);
    driver.set_virtualizable_info(std::sync::Arc::new(VirtualizableInfo::new(0)));
    let mut state = VirtualizableSyncState {
        frame: 1,
        stack: 2,
        before_sync_calls: 0,
        after_sync_calls: 0,
        reject_sync: true,
    };

    assert!(
        driver
            .back_edge_keyed(77, 77, &mut state, &(), || {})
            .is_none()
    );
    assert!(
        driver
            .back_edge_keyed(77, 77, &mut state, &(), || {})
            .is_none()
    );
    // `warmstate.py maybe_compile_and_run` ticks before `bound_reached`: the
    // cold edge does not build/sync frontend state, while the hot edge still
    // gives `sync_before` the same veto over the tracing start.
    assert_eq!(state.before_sync_calls, 1);
    assert_eq!(state.after_sync_calls, 0);
    assert!(!driver.is_tracing());
}

#[test]
fn jit_state_restore_guard_failure_restores_from_reconstructed_resume_frame() {
    let meta = TestMeta { header_pc: 444 };
    let frame_ptr = 0x4444usize;
    let mut state = ResumeMappedState {
        frame: 0,
        stackpos: 1,
        top: -10,
    };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 444);
    resume.set_slot_constant(0, majit_ir::Const::from_gcref(GcRef(frame_ptr)));
    resume.map_slot(1, 0);
    resume.set_slot_constant(2, majit_ir::Const::Int(99));
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.frame, frame_ptr);
    assert_eq!(state.stackpos, 2);
    assert_eq!(state.top, 99);
}

#[test]
fn jit_state_restore_guard_failure_materializes_virtual_ref_from_resume_state() {
    let meta = TestMeta { header_pc: 555 };
    let typedescr_7: majit_ir::DescrRef =
        std::sync::Arc::new(majit_ir::descr::SimpleSizeDescr::new(7, 0, 0));
    let mut state = VirtualResumeState {
        obj: 0,
        flag: 1,
        materialized_ref: 0xfeedusize,
        materialize_calls: 0,
        expected_descr: Some(typedescr_7.clone()),
    };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 555);
    let virtual_index = resume.add_virtual_struct(
        Some(typedescr_7),
        0,
        vec![(
            3,
            majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::Int(55)),
        )],
        vec![],
        0,
    );
    resume.set_slot_virtual(0, virtual_index);
    resume.map_slot(1, 0);
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.obj, 0xfeedusize);
    assert_eq!(state.flag, 2);
    assert_eq!(state.materialize_calls, 1);
}

#[test]
fn jit_state_restore_guard_failure_materializes_nested_virtual_refs_in_dependency_order() {
    let inner_typedescr: majit_ir::DescrRef =
        std::sync::Arc::new(majit_ir::descr::SimpleSizeDescr::new(10, 0, 0));
    let outer_typedescr: majit_ir::DescrRef =
        std::sync::Arc::new(majit_ir::descr::SimpleSizeDescr::new(20, 0, 0));
    let mut state = NestedVirtualResumeState {
        outer: 0,
        inner: 0,
        alias: 0,
        outer_ref: 0xbeefusize,
        inner_ref: 0xabbausize,
        materialize_order: Vec::new(),
        expected_inner_descr: Some(inner_typedescr.clone()),
        expected_outer_descr: Some(outer_typedescr.clone()),
    };
    let meta = TestMeta { header_pc: 556 };
    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 556);
    let inner = resume.add_virtual_struct(
        Some(inner_typedescr),
        0,
        vec![(
            0,
            majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::Int(77)),
        )],
        vec![],
        0,
    );
    let outer = resume.add_virtual_obj(
        Some(outer_typedescr),
        0,
        None,
        vec![
            (
                0,
                majit_metainterp::resume::ResumeValueSource::Virtual(inner),
            ),
            (
                1,
                majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::Int(99)),
            ),
        ],
        vec![],
        0,
    );
    resume.set_slot_virtual(0, outer);
    resume.set_slot_virtual(1, inner);
    resume.set_slot_virtual(2, inner);
    let reconstructed_state = resume.build().reconstruct_state(&[]);

    assert!(state.restore_guard_failure(
        &meta,
        &[],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &[],
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.outer, state.outer_ref);
    assert_eq!(state.inner, state.inner_ref);
    assert_eq!(state.alias, state.inner_ref);
    assert_eq!(state.materialize_order, vec![0, 1]);
}

#[test]
fn jit_state_restore_guard_failure_replays_pending_writes_with_virtual_target_and_value() {
    let mut parent_cell = Box::new(PendingRefCell { child: 0 });
    let mut child_cell = Box::new(PendingRefCell { child: 0 });
    let parent_ref = (&mut *parent_cell as *mut PendingRefCell) as usize;
    let child_ref = (&mut *child_cell as *mut PendingRefCell) as usize;
    let parent_typedescr: majit_ir::DescrRef =
        std::sync::Arc::new(majit_ir::descr::SimpleSizeDescr::new(30, 0, 0));
    let child_typedescr: majit_ir::DescrRef =
        std::sync::Arc::new(majit_ir::descr::SimpleSizeDescr::new(31, 0, 0));
    let mut state = PendingVirtualWriteState {
        parent: 0,
        child: 0,
        parent_ref,
        child_ref,
        materialize_order: Vec::new(),
        expected_parent_descr: Some(parent_typedescr.clone()),
        expected_child_descr: Some(child_typedescr.clone()),
    };
    let meta = TestMeta { header_pc: 557 };
    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 557);
    let parent = resume.add_virtual_struct(Some(parent_typedescr), 0, vec![], vec![], 0);
    let child = resume.add_virtual_struct(Some(child_typedescr), 0, vec![], vec![], 0);
    resume.set_slot_virtual(0, parent);
    resume.set_slot_virtual(1, child);
    let pending_descr: majit_ir::DescrRef = std::sync::Arc::new(
        majit_ir::descr::SimpleFieldDescr::new(21, 0, 8, Type::Ref, false),
    );
    resume.add_pending_field_write(
        Some(pending_descr),
        majit_metainterp::resume::ResumeValueSource::Virtual(parent),
        majit_metainterp::resume::ResumeValueSource::Virtual(child),
    );
    let reconstructed_state = resume.build().reconstruct_state(&[]);

    assert!(state.restore_guard_failure(
        &meta,
        &[],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.parent, parent_ref);
    assert_eq!(state.child, child_ref);
    assert_eq!(parent_cell.child, child_ref);
    assert_eq!(state.materialize_order, vec![0, 1]);
}

#[test]
fn jit_state_restore_guard_failure_replays_pending_field_writes() {
    let meta = TestMeta { header_pc: 666 };
    let mut cell = PendingWriteCell { field: 11 };
    let cell_ptr = (&mut cell as *mut PendingWriteCell) as usize;
    let mut state = PendingWriteState { obj: 0, flag: 1 };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 666);
    resume.set_slot_constant(0, majit_ir::Const::from_gcref(GcRef(cell_ptr)));
    resume.map_slot(1, 0);
    let pending_descr: majit_ir::DescrRef = std::sync::Arc::new(
        majit_ir::descr::SimpleFieldDescr::new(9, 0, 8, Type::Int, false),
    );
    resume.add_pending_field_write(
        Some(pending_descr),
        majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::from_gcref(GcRef(
            cell_ptr,
        ))),
        majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::Int(77)),
    );
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.flag, 2);
    assert_eq!(cell.field, 77);
}

#[test]
fn jit_state_restore_guard_failure_replays_pending_array_writes_via_layout_hook() {
    let meta = TestMeta { header_pc: 888 };
    let mut array = vec![10_i64, 20_i64, 30_i64];
    let array_ptr = array.as_mut_ptr() as usize;
    let mut state = PendingArrayWriteState { array: 0, flag: 1 };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 888);
    resume.set_slot_constant(0, majit_ir::Const::from_gcref(GcRef(array_ptr)));
    resume.map_slot(1, 0);
    let pending_descr: majit_ir::DescrRef = std::sync::Arc::new(
        majit_ir::descr::SimpleArrayDescr::new(12, 0, std::mem::size_of::<i64>(), 0, Type::Int),
    );
    resume.add_pending_arrayitem_write(
        Some(pending_descr),
        majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::from_gcref(GcRef(
            array_ptr,
        ))),
        1,
        majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::Int(88)),
    );
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.flag, 2);
    assert_eq!(array, vec![10, 88, 30]);
}

#[test]
fn jit_state_restore_guard_failure_can_restore_multi_frame_resume_state() {
    let meta = TestMeta { header_pc: 777 };
    let frame_ptr = 0xabcusize;
    let mut state = MultiFrameResumeState {
        frame: 0,
        flag: 1,
        restored_pcs: Vec::new(),
    };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 100);
    resume.set_slot_constant(0, majit_ir::Const::from_gcref(GcRef(frame_ptr)));
    resume.set_slot_constant(1, majit_ir::Const::Int(1));
    resume.push_frame(0, 200);
    resume.set_slot_constant(0, majit_ir::Const::from_gcref(GcRef(frame_ptr)));
    resume.map_slot(1, 0);
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.frame, frame_ptr);
    assert_eq!(state.flag, 2);
    assert_eq!(state.restored_pcs, vec![100, 200]);
}

#[test]
fn jit_state_restore_guard_failure_can_restore_multi_frame_state_via_generic_frame_hooks() {
    let meta = TestMeta { header_pc: 778 };
    let materialized_ref = 0xfeedusize;
    let mut state = GenericMultiFrameResumeState {
        frame: 0,
        flag: 1,
        restored_pcs: Vec::new(),
        restored_frames: Vec::new(),
        materialized_ref,
        materialize_calls: 0,
    };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 300);
    let virtual_index = resume.add_virtual_struct(None, 0, vec![], vec![], 0);
    resume.set_slot_virtual(0, virtual_index);
    resume.set_slot_constant(1, majit_ir::Const::Int(1));
    resume.push_frame(0, 400);
    resume.set_slot_virtual(0, virtual_index);
    resume.map_slot(1, 0);
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.frame, materialized_ref);
    assert_eq!(state.flag, 2);
    assert_eq!(state.restored_pcs, vec![300, 400]);
    assert_eq!(
        state.restored_frames,
        vec![(materialized_ref, 1), (materialized_ref, 2)]
    );
    assert_eq!(state.materialize_calls, 1);
}

#[test]
fn jit_state_restore_guard_failure_reuses_virtual_cache_for_pending_writes() {
    let meta = TestMeta { header_pc: 779 };
    let mut cell = PendingWriteCell { field: 0 };
    let cell_ptr = (&mut cell as *mut PendingWriteCell) as usize;
    let mut state = GenericMultiFrameResumeState {
        frame: 0,
        flag: 1,
        restored_pcs: Vec::new(),
        restored_frames: Vec::new(),
        materialized_ref: cell_ptr,
        materialize_calls: 0,
    };

    let mut resume = ResumeDataVirtualAdder::new();
    resume.push_frame(0, 500);
    let virtual_index = resume.add_virtual_struct(None, 0, vec![], vec![], 0);
    resume.set_slot_virtual(0, virtual_index);
    resume.set_slot_constant(1, majit_ir::Const::Int(1));
    resume.push_frame(0, 600);
    resume.set_slot_virtual(0, virtual_index);
    resume.map_slot(1, 0);
    let pending_descr: majit_ir::DescrRef = std::sync::Arc::new(
        majit_ir::descr::SimpleFieldDescr::new(31, 0, 8, Type::Int, false),
    );
    resume.add_pending_field_write(
        Some(pending_descr),
        majit_metainterp::resume::ResumeValueSource::Virtual(virtual_index),
        majit_metainterp::resume::ResumeValueSource::Constant(majit_ir::Const::Int(77)),
    );
    let reconstructed_state = resume.build().reconstruct_state(&[2]);

    assert!(state.restore_guard_failure(
        &meta,
        &[2],
        Some(&reconstructed_state),
        &reconstructed_state.virtuals,
        &reconstructed_state.pending_fields,
        &majit_metainterp::blackhole::ExceptionState::default(),
    ));

    assert_eq!(state.materialize_calls, 1);
    assert_eq!(cell.field, 77);
}
