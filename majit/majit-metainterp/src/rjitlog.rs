//! `rpython/rlib/rjitlog/rjitlog.py`.
//!
//! Binary jitlog writer. `JITLOG` names the output file
//! (`rjitlog.c jitlog_try_init_using_env`). The trace id increments even
//! when the file is unset.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Mutex;

use majit_ir::operand::Operand;
use majit_ir::{InputArg, Op, OpCode, Type, Value};

/// `rjitlog.py` mark table, `start = 0x11`.
const MARK_BASE: u8 = 0x11;
pub const MARK_INPUT_ARGS: u8 = MARK_BASE;
pub const MARK_RESOP_META: u8 = MARK_BASE + 1;
pub const MARK_RESOP: u8 = MARK_BASE + 2;
pub const MARK_RESOP_DESCR: u8 = MARK_BASE + 3;
pub const MARK_TRACE: u8 = MARK_BASE + 6;
pub const MARK_TRACE_OPT: u8 = MARK_BASE + 7;
pub const MARK_STITCH_BRIDGE: u8 = MARK_BASE + 9;
pub const MARK_START_TRACE: u8 = MARK_BASE + 10;
pub const MARK_INIT_MERGE_POINT: u8 = MARK_BASE + 12;
pub const MARK_JITLOG_HEADER: u8 = MARK_BASE + 13;
pub const MARK_MERGE_POINT: u8 = MARK_BASE + 14;
pub const MARK_COMMON_PREFIX: u8 = MARK_BASE + 15;
pub const MARK_ABORT_TRACE: u8 = MARK_BASE + 16;
pub const MARK_REDIRECT_ASSEMBLER: u8 = MARK_BASE + 18;
pub const MARK_TMP_CALLBACK: u8 = MARK_BASE + 19;

/// `rjitlog.py` MP_* semantic types.
pub const MP_STR: u8 = 0x0;
pub const MP_INT: u8 = 0x0;
pub const MP_FILENAME: u8 = 0x1;
pub const MP_LINENO: u8 = 0x2;
pub const MP_INDEX: u8 = 0x4;
pub const MP_SCOPE: u8 = 0x8;
pub const MP_OPCODE: u8 = 0x10;

/// `rjitlog.py JITLOG_VERSION`.
const JITLOG_VERSION: u16 = 4;

/// `rjitlog.py wrap` result for one merge-point field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MpValue {
    Str { sem: u8, value: String },
    Int { sem: u8, value: i64 },
}

pub type GetLocation = fn(&[i64]) -> Option<Vec<MpValue>>;

struct JitLogState {
    file: Option<File>,
    trace_id: u64,
    /// Snapshot of `metainterp_sd` addr2name at `start_new_trace`.
    /// `rjitlog.py JitLogger.start_new_trace` stores `self.metainterp_sd`.
    addr2name: Vec<(u64, String)>,
    /// `warmstate.py get_location` / `get_location_types`.
    location_types: Vec<(u8, u8)>,
    get_location: Option<GetLocation>,
}

static JITLOG: Mutex<JitLogState> = Mutex::new(JitLogState {
    file: None,
    trace_id: 0,
    addr2name: Vec::new(),
    location_types: Vec::new(),
    get_location: None,
});

fn lock() -> std::sync::MutexGuard<'static, JitLogState> {
    JITLOG.lock().unwrap_or_else(|e| e.into_inner())
}

/// `rjitlog.c jitlog_try_init_using_env`.
pub fn jitlog_try_init_using_env() {
    let Some(path) = std::env::var_os("JITLOG") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let mut state = lock();
    if state.file.is_some() {
        return;
    }
    match File::create(&path) {
        Ok(file) => state.file = Some(file),
        Err(err) => eprintln!("could not open '{}': {err}", path.to_string_lossy()),
    }
}

/// `rjitlog.py JitLogger.setup_once`, run from `_setup_once`.
pub fn setup_once() {
    if jitlog_enabled() {
        return;
    }
    jitlog_try_init_using_env();
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let header = assemble_header();
    write_marked(&mut state, MARK_JITLOG_HEADER, &header);
}

/// `rjitlog.py assemble_header`.
fn assemble_header() -> Vec<u8> {
    let mut content = Vec::new();
    content.extend_from_slice(&JITLOG_VERSION.to_le_bytes());
    content.push(0); // 64-bit
    content.extend_from_slice(&encode_str(std::env::consts::ARCH));
    content.push(MARK_RESOP_META);
    let opcodes: Vec<OpCode> = OpCode::all().collect();
    content.extend_from_slice(&encode_le_16bit(opcodes.len() as u16));
    for op in opcodes {
        content.extend_from_slice(&encode_le_16bit(op.as_u16()));
        content.extend_from_slice(&encode_str(&op.name().to_ascii_lowercase()));
    }
    content
}

/// `rjitlog.c jitlog_enabled`.
pub fn jitlog_enabled() -> bool {
    lock().file.is_some()
}

/// `rjitlog.py JitLogger.start_new_trace`. Increments even when disabled.
///
/// `descr_or_entry` is `compute_unique_id(faildescr)` for a bridge, or
/// `int(entry_bridge)` for a loop.
pub fn start_new_trace(is_bridge: bool, descr_or_entry: u64, jd_name: &str) -> u64 {
    let mut state = lock();
    state.trace_id += 1;
    let tid = state.trace_id;
    if state.file.is_some() {
        let kind = if is_bridge { "bridge" } else { "loop" };
        let mut payload = encode_le_addr(tid).to_vec();
        payload.extend_from_slice(&encode_str(kind));
        payload.extend_from_slice(&encode_le_addr(descr_or_entry));
        payload.extend_from_slice(&encode_str(jd_name));
        write_marked(&mut state, MARK_START_TRACE, &payload);
    }
    tid
}

/// `rjitlog.py JitLogger.start_new_trace`: `self.metainterp_sd = metainterp_sd`.
/// Snapshot `addr2name` so later `var_to_str` can emit `ConstClass(name)`.
pub fn set_addr2name(names: impl IntoIterator<Item = (u64, String)>) {
    lock().addr2name = names.into_iter().collect();
}

/// `warmstate.py`: `self.get_location` / `self.get_location_types`.
///
/// `types` is the `@returns(MP_*)` list: `(semantic, generic)` where
/// generic is `b's'` or `b'i'`.
pub fn register_get_location(types: &[(u8, u8)], get_location: GetLocation) {
    let mut state = lock();
    state.location_types = types.to_vec();
    state.get_location = Some(get_location);
}

/// `rpython.rlib.rjitlog.redirect_assembler` — assemblers call
/// `majit_backend::redirect_assembler`; this is the writer they install.
pub fn install_backend_hooks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        majit_backend::register_redirect_assembler_log(redirect_assembler);
    });
}

/// `rjitlog.py redirect_assembler`.
pub fn redirect_assembler(old_id: u64, new_id: u64, asm_adr: u64) {
    install_backend_hooks();
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let mut payload = encode_le_addr(old_id).to_vec();
    payload.extend_from_slice(&encode_le_addr(new_id));
    payload.extend_from_slice(&encode_le_addr(asm_adr));
    write_marked(&mut state, MARK_REDIRECT_ASSEMBLER, &payload);
}

/// `rjitlog.py JitLogger.log_patch_guard`.
pub fn log_patch_guard(descr_number: u64, addr: u64) {
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let mut payload = encode_le_addr(descr_number).to_vec();
    payload.extend_from_slice(&encode_le_addr(addr));
    write_marked(&mut state, MARK_STITCH_BRIDGE, &payload);
}

/// `rjitlog.py tmp_callback`.
pub fn tmp_callback(token_id: u64, token_number: u64) {
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let mut payload = encode_le_addr(token_id).to_vec();
    payload.extend_from_slice(&encode_le_addr(token_number));
    write_marked(&mut state, MARK_TMP_CALLBACK, &payload);
}

/// `rjitlog.py JitLogger.trace_aborted`.
pub fn trace_aborted(tid: u64) {
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let payload = encode_le_addr(tid);
    write_marked(&mut state, MARK_ABORT_TRACE, &payload);
}

/// Current `JitLogger.trace_id`.
pub fn trace_id() -> u64 {
    lock().trace_id
}

/// `rjitlog.py JitLogger.log_trace` + `LogTrace.write`.
///
/// `tid` is the id `start_new_trace` returned for this compile, not the
/// process-global latest id (`JitLogger.trace_id` on the logger that
/// started the write).
pub fn write_trace<A, O>(tag: u8, tid: u64, inputargs: &[A], ops: &[O])
where
    A: Borrow<InputArg>,
    O: Borrow<Op>,
{
    let mut state = lock();
    if state.file.is_none() {
        return;
    }
    let header = encode_le_addr(tid);
    write_marked(&mut state, tag, &header);
    let mut memo = VarMemo {
        addr2name: state.addr2name.iter().cloned().collect(),
        ..VarMemo::default()
    };
    let args: Vec<String> = inputargs
        .iter()
        .map(|arg| memo.inputarg(arg.borrow()))
        .collect();
    write_marked(&mut state, MARK_INPUT_ARGS, &encode_str(&args.join(",")));
    let get_location = state.get_location;
    let location_types = state.location_types.clone();
    let mut compressor: Option<PrefixCompressor> = None;
    for op in ops {
        let op = op.borrow();
        if op.opcode == OpCode::DebugMergePoint {
            encode_debug_info(
                &mut state,
                &mut compressor,
                get_location,
                &location_types,
                op,
            );
            continue;
        }
        write_resop(&mut state, &mut memo, op);
    }
}

/// `rjitlog.py LogTrace.var_to_str` memo.
#[derive(Default)]
struct VarMemo {
    next: usize,
    ids: HashMap<(u8, u64), usize>,
    addr2name: HashMap<u64, String>,
}

impl VarMemo {
    fn assign(&mut self, kind: u8, key: u64) -> usize {
        *self.ids.entry((kind, key)).or_insert_with(|| {
            let id = self.next;
            self.next += 1;
            id
        })
    }

    fn inputarg(&mut self, arg: &InputArg) -> String {
        let kind = match arg.tp.get() {
            Type::Int => b'i',
            Type::Ref => b'p',
            Type::Float => b'f',
            Type::Void => b'?',
        };
        let id = self.assign(kind, arg.index as u64);
        format!("{}{id}", kind as char)
    }

    fn op_result(&mut self, op: &Op) -> String {
        // rjitlog.py var_to_str: void / unknown type is `?`, and the
        // memo slot is still allocated.
        let kind = match op.type_ {
            Type::Int => b'i',
            Type::Ref => b'p',
            Type::Float => b'f',
            Type::Void => b'?',
        };
        let id = self.assign(kind, op.pos().get().raw() as u64);
        if op.type_ == Type::Void {
            return "?".into();
        }
        format!("{}{id}", kind as char)
    }

    /// `rjitlog.py LogTrace.var_to_str`.
    fn operand(&mut self, arg: &Operand) -> String {
        if arg.is_none() {
            return "-".into();
        }
        if let Some(value) = arg.const_value() {
            // rjitlog.py var_to_str: memo is keyed by the box object,
            // not the printed value. Value::Void is not an upstream
            // constant; do not consume a slot.
            if matches!(value, Value::Void) {
                return "None".into();
            }
            let id = self.assign(b'k', arg.identity_key());
            return match value {
                Value::Int(v) => {
                    // rjitlog.py var_to_str: ConstClass(name) when the
                    // int could be an address and addr2name hits.
                    if int_could_be_an_address(v)
                        && let Some(name) = self.addr2name.get(&(v as u64))
                        && !name.is_empty()
                    {
                        return format!("ConstClass({name})");
                    }
                    v.to_string()
                }
                Value::Float(v) => python_float_str(v),
                Value::Ref(r) if r.is_null() => "ConstPtr(null)".into(),
                Value::Ref(_) => format!("ConstPtr(ptr{id})"),
                Value::Void => "None".into(),
            };
        }
        if arg.is_null_ref() {
            let _ = self.assign(b'k', arg.identity_key());
            return "ConstPtr(null)".into();
        }
        if arg.is_inputarg() {
            let opref = arg.to_opref();
            let tp = arg.type_();
            let idx = opref.raw() as u64;
            let kind = match tp {
                Type::Int => b'i',
                Type::Ref => b'p',
                Type::Float => b'f',
                Type::Void => b'?',
            };
            let id = self.assign(kind, idx);
            return format!("{}{id}", kind as char);
        }
        if arg.is_resop() {
            let kind = match arg.type_() {
                Type::Int => b'i',
                Type::Ref => b'p',
                Type::Float => b'f',
                Type::Void => b'v',
            };
            let id = self.assign(kind, arg.to_opref().raw() as u64);
            return format!("{}{id}", kind as char);
        }
        "?".into()
    }
}

/// `rjitlog.py LogTrace.encode_debug_info`.
fn encode_debug_info(
    state: &mut JitLogState,
    compressor: &mut Option<PrefixCompressor>,
    get_location: Option<GetLocation>,
    types: &[(u8, u8)],
    op: &Op,
) {
    let Some(get_location) = get_location else {
        return;
    };
    if types.is_empty() {
        return;
    }
    let args = op.getarglist();
    if args.len() < 3 {
        return;
    }
    let greens: Vec<i64> = args[3..].iter().filter_map(operand_green_int).collect();
    let Some(values) = get_location(&greens) else {
        return;
    };
    if compressor.is_none() {
        let mut encoded_types = encode_le_16bit(types.len() as u16).to_vec();
        for &(sem, generic) in types {
            encoded_types.push(sem);
            encoded_types.push(generic);
        }
        write_marked(state, MARK_INIT_MERGE_POINT, &encoded_types);
        *compressor = Some(PrefixCompressor::new(types.len()));
    }
    let encoded = encode_merge_point(
        |mark, payload| write_marked(state, mark, payload),
        compressor.as_mut().unwrap(),
        &values,
    );
    write_marked(state, MARK_MERGE_POINT, &encoded);
}

fn operand_green_int(arg: &Operand) -> Option<i64> {
    match arg.const_value()? {
        Value::Int(v) => Some(v),
        Value::Ref(r) => Some(r.0 as i64),
        _ => None,
    }
}

/// `rjitlog.py PrefixCompressor`.
struct PrefixCompressor {
    prefixes: Vec<Option<String>>,
    written_prefixes: Vec<Option<String>>,
}

impl PrefixCompressor {
    fn new(count: usize) -> Self {
        Self {
            prefixes: vec![None; count],
            written_prefixes: vec![None; count],
        }
    }

    fn get_last(&self, index: usize) -> Option<&str> {
        self.prefixes.get(index).and_then(|s| s.as_deref())
    }

    fn get_last_written(&self, index: usize) -> Option<&str> {
        self.written_prefixes.get(index).and_then(|s| s.as_deref())
    }

    fn compress(&mut self, index: usize, string: &str) -> Option<String> {
        let last = self.get_last(index);
        if last.is_none() {
            self.prefixes[index] = Some(string.to_string());
            return None;
        }
        let cp = commonprefix(last.unwrap(), string);
        if cp.len() <= 1 {
            self.prefixes[index] = Some(string.to_string());
            return None;
        }
        Some(cp)
    }

    fn write_prefix<W: FnMut(u8, &[u8])>(
        &mut self,
        mut write_marked: W,
        index: usize,
        prefix: &str,
    ) {
        let mut payload = Vec::with_capacity(1 + 4 + prefix.len());
        payload.push(index as u8);
        payload.extend_from_slice(&encode_str(prefix));
        write_marked(MARK_COMMON_PREFIX, &payload);
        self.written_prefixes[index] = Some(prefix.to_string());
    }
}

/// `rjitlog.py commonprefix`.
fn commonprefix(a: &str, b: &str) -> String {
    let n = a.len().min(b.len());
    let bytes_a = a.as_bytes();
    let bytes_b = b.as_bytes();
    let mut i = 0;
    while i < n && bytes_a[i] == bytes_b[i] {
        i += 1;
    }
    // Stay on a char boundary.
    while i > 0 && !a.is_char_boundary(i) {
        i -= 1;
    }
    a[..i].to_string()
}

/// `rjitlog.py encode_merge_point`.
fn encode_merge_point<W: FnMut(u8, &[u8])>(
    mut write_marked: W,
    compressor: &mut PrefixCompressor,
    values: &[MpValue],
) -> Vec<u8> {
    let mut line = Vec::new();
    for (i, value) in values.iter().enumerate() {
        match value {
            MpValue::Int { value, .. } => {
                line.push(0x00);
                line.extend_from_slice(&encode_le_addr(*value as u64));
            }
            MpValue::Str { value, .. } => {
                let last_prefix = compressor.get_last_written(i).map(str::to_string);
                let cp = compressor.compress(i, value);
                match cp {
                    None => {
                        line.push(0xff);
                        line.extend_from_slice(&encode_str(value));
                    }
                    Some(cp) => {
                        if last_prefix.as_deref() != Some(cp.as_str()) {
                            compressor.write_prefix(&mut write_marked, i, &cp);
                        }
                        if value.len() == cp.len() {
                            line.push(0xef);
                        } else {
                            line.push(0x00);
                            line.extend_from_slice(&encode_str(&value[cp.len()..]));
                        }
                    }
                }
            }
        }
    }
    line
}

/// `rjitlog.py LogTrace.encode_op`.
fn write_resop(state: &mut JitLogState, memo: &mut VarMemo, op: &Op) {
    let descr = op.getdescr();
    let mark = if descr.is_some() {
        MARK_RESOP_DESCR
    } else {
        MARK_RESOP
    };
    let mut line = encode_le_16bit(op.opcode.as_u16()).to_vec();
    // rjitlog.py encode_op: memoize arguments, then the result.
    let arg_strs: Vec<String> = op
        .getarglist()
        .iter()
        .map(|arg| memo.operand(arg))
        .collect();
    let mut body = memo.op_result(op);
    for arg in arg_strs {
        body.push(',');
        body.push_str(&arg);
    }
    if let Some(ref d) = descr {
        // rjitlog.py encode_op: descr.repr_of_descr().
        body.push(',');
        body.push_str(&d.repr());
    }
    line.extend_from_slice(&encode_str(&body));
    if let Some(d) = descr {
        line.extend_from_slice(&encode_le_addr(
            std::sync::Arc::as_ptr(&d) as *const () as u64
        ));
    }
    let failargs = match op.getfailargs() {
        Some(args) => args
            .iter()
            .map(|arg| memo.operand(arg))
            .collect::<Vec<_>>()
            .join(","),
        None => String::new(),
    };
    line.extend_from_slice(&encode_str(&failargs));
    write_marked(state, mark, &line);
}

fn write_marked(state: &mut JitLogState, mark: u8, payload: &[u8]) {
    let Some(file) = state.file.as_mut() else {
        return;
    };
    let mut buf = Vec::with_capacity(1 + payload.len());
    buf.push(mark);
    buf.extend_from_slice(payload);
    let _ = file.write_all(&buf);
}

/// `rjitlog.py encode_le_16bit`.
pub fn encode_le_16bit(val: u16) -> [u8; 2] {
    val.to_le_bytes()
}

/// `rjitlog.py encode_le_addr` on 64-bit.
pub fn encode_le_addr(val: u64) -> [u8; 8] {
    val.to_le_bytes()
}

/// `rjitlog.py int_could_be_an_address` after translation.
fn int_could_be_an_address(x: i64) -> bool {
    !(-32768..=32767).contains(&x)
}

/// `str(float)` spelling from `rjitlog.py var_to_str` (`str(arg.getfloat())`).
///
/// Rust `f64::to_string()` writes `NaN`, omits `.0` on integrals, and keeps
/// fixed form past Python's `1e16` / `1e-4` switch. Match CPython's
/// lowercase non-finites, signed two-digit exponents, and those thresholds.
fn python_float_str(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if !v.is_finite() {
        return if v.is_sign_negative() {
            "-inf".into()
        } else {
            "inf".into()
        };
    }
    if v == 0.0 {
        return if v.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let abs = v.abs();
    if abs >= 1e16 || abs < 1e-4 {
        return python_scientific_str(v);
    }
    let s = v.to_string();
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

/// `str(float)` scientific form. rustc's `{:e}` is the correctly-rounded
/// shortest mantissa; Python only differs in the exponent's sign and
/// minimum width (`e+16`, `e-05`). Reconstructing `mant * 10**exp` in
/// `f64` underflows (`5e-324` → `infe-323`) and changes trailing digits.
fn python_scientific_str(v: f64) -> String {
    let s = format!("{v:e}");
    let Some((mant, exp)) = s.split_once('e') else {
        return s;
    };
    let Ok(exp) = exp.parse::<i32>() else {
        return s;
    };
    format!("{mant}e{exp:+03}")
}

/// `rjitlog.py encode_str`.
pub fn encode_str(string: &str) -> Vec<u8> {
    let len = string.len() as u32;
    let mut out = Vec::with_capacity(4 + string.len());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(string.as_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_trace_is_the_seventh_mark() {
        assert_eq!(MARK_TRACE, 0x17);
        assert_eq!(MARK_STITCH_BRIDGE, 0x1a);
        assert_eq!(MARK_START_TRACE, 0x1b);
        assert_eq!(MARK_INIT_MERGE_POINT, 0x1d);
        assert_eq!(MARK_JITLOG_HEADER, 0x1e);
        assert_eq!(MARK_MERGE_POINT, 0x1f);
        assert_eq!(MARK_COMMON_PREFIX, 0x20);
        assert_eq!(MARK_ABORT_TRACE, 0x21);
        assert_eq!(MARK_REDIRECT_ASSEMBLER, 0x23);
        assert_eq!(MARK_TMP_CALLBACK, 0x24);
    }

    #[test]
    fn encode_str_prefixes_little_endian_length() {
        let encoded = encode_str("ab");
        assert_eq!(&encoded[..4], &[2, 0, 0, 0]);
        assert_eq!(&encoded[4..], b"ab");
    }

    #[test]
    fn start_new_trace_increments_when_disabled() {
        let before = trace_id();
        let tid = start_new_trace(false, 0, "test");
        assert!(tid > before);
    }

    #[test]
    fn write_trace_is_a_no_op_when_disabled() {
        write_trace::<InputArg, Op>(MARK_TRACE, 0, &[], &[]);
        write_trace::<InputArg, Op>(MARK_TRACE_OPT, 0, &[], &[]);
        redirect_assembler(0, 1, 2);
        log_patch_guard(0, 0);
        tmp_callback(0, 0);
    }

    #[test]
    fn var_to_str_names_inputargs_i0_p1() {
        let mut memo = VarMemo::default();
        let i = InputArg::new_int(0);
        let p = InputArg::from_type(Type::Ref, 1);
        assert_eq!(memo.inputarg(&i), "i0");
        assert_eq!(memo.inputarg(&p), "p1");
        assert_eq!(memo.inputarg(&i), "i0");
    }

    #[test]
    fn void_result_is_question_mark() {
        let mut memo = VarMemo::default();
        let op = Op::new(OpCode::Finish, &[]);
        assert_eq!(memo.op_result(&op), "?");
    }

    #[test]
    fn const_operands_reserve_a_memo_slot() {
        let mut memo = VarMemo::default();
        assert_eq!(memo.operand(&Operand::none()), "-");
        assert_eq!(memo.operand(&Operand::const_from_value(Value::Int(5))), "5");
        assert_eq!(
            memo.operand(&Operand::const_from_value(Value::Ref(majit_ir::GcRef(
                0x1000
            )))),
            "ConstPtr(ptr1)"
        );
        // ConstInt reserved slot 0; ConstPtr reserved slot 1; the next
        // named box is 2. Value::Void must not steal a slot.
        assert_eq!(
            memo.operand(&Operand::const_from_value(Value::Void)),
            "None"
        );
        let i = InputArg::new_int(0);
        assert_eq!(memo.inputarg(&i), "i2");
    }

    #[test]
    fn const_class_uses_addr2name() {
        let mut memo = VarMemo::default();
        memo.addr2name.insert(0x10000, "alpha".into());
        assert_eq!(
            memo.operand(&Operand::const_from_value(Value::Int(0x10000))),
            "ConstClass(alpha)"
        );
        assert_eq!(memo.operand(&Operand::const_from_value(Value::Int(5))), "5");
    }

    #[test]
    fn float_constants_keep_python_spelling() {
        assert_eq!(python_float_str(1.0), "1.0");
        assert_eq!(python_float_str(-0.0), "-0.0");
        assert_eq!(python_float_str(0.0), "0.0");
        assert_eq!(python_float_str(1.5), "1.5");
        assert_eq!(python_float_str(1e16), "1e+16");
        assert_eq!(python_float_str(1e15), "1000000000000000.0");
        assert_eq!(python_float_str(1e-4), "0.0001");
        assert_eq!(python_float_str(1e-5), "1e-05");
        assert_eq!(python_float_str(1e20), "1e+20");
        assert_eq!(python_float_str(5e-324), "5e-324");
        assert_eq!(python_float_str(-5e-324), "-5e-324");
        assert_eq!(python_float_str(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(python_float_str(f64::MIN), "-1.7976931348623157e+308");
        assert_eq!(
            python_float_str(2.2250738585072014e-308),
            "2.2250738585072014e-308"
        );
        assert_eq!(python_float_str(f64::INFINITY), "inf");
        assert_eq!(python_float_str(f64::NEG_INFINITY), "-inf");
        assert_eq!(python_float_str(f64::NAN), "nan");
        let mut memo = VarMemo::default();
        assert_eq!(
            memo.operand(&Operand::const_from_value(Value::Float(1.0))),
            "1.0"
        );
    }

    #[test]
    fn equal_const_ints_keep_separate_memo_slots() {
        let mut memo = VarMemo::default();
        let a = Operand::const_from_value(Value::Int(5));
        let b = Operand::const_from_value(Value::Int(5));
        assert_eq!(memo.operand(&a), "5");
        assert_eq!(memo.operand(&b), "5");
        let i = InputArg::new_int(0);
        assert_eq!(memo.inputarg(&i), "i2");
    }

    #[test]
    fn common_prefix_matches_upstream() {
        assert_eq!(commonprefix("", ""), "");
        assert_eq!(commonprefix("/hello/world", "/path/to"), "/");
        assert_eq!(commonprefix("pyramid", "python"), "py");
        assert_eq!(
            commonprefix(&"0".repeat(100), &"0".repeat(100)),
            "0".repeat(100)
        );

        let mut prefixes = Vec::new();
        let mut compressor = PrefixCompressor::new(1);
        let hello = MpValue::Str {
            sem: 0,
            value: "hello".into(),
        };
        let result = encode_merge_point(
            |mark, payload| prefixes.push((mark, payload.to_vec())),
            &mut compressor,
            &[hello.clone()],
        );
        assert_eq!(result[0], 0xff);
        assert_eq!(&result[1..], &encode_str("hello"));
        assert!(prefixes.is_empty());

        let result = encode_merge_point(
            |mark, payload| prefixes.push((mark, payload.to_vec())),
            &mut compressor,
            &[hello],
        );
        assert_eq!(result, vec![0xef]);
        assert_eq!(prefixes.len(), 1);
        assert_eq!(prefixes[0].0, MARK_COMMON_PREFIX);
        assert_eq!(prefixes[0].1[0], 0);
        assert_eq!(&prefixes[0].1[1..], &encode_str("hello"));
    }
}
