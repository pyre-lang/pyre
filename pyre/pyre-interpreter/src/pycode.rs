//! PyCode — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::nestedscope::CellFamily;
use pyre_object::pyobject::*;
use pyre_object::{
    FixedObjectArray, w_bool_from, w_bool_get_value, w_int_new, w_list_new, w_seq_iter_new,
    w_str_new, w_str_new_managed, w_tuple_new,
};
use rustpython_compiler_core::SourceLocation;
use rustpython_compiler_core::bytecode::{Constants, PyCodeLocationInfoKind};

const YIELDS_INSIDE_TRY_BIT: u16 = 0x8000;

/// Compatibility marker for malformed bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeCorruption;

impl From<BytecodeCorruption> for crate::PyError {
    #[inline(always)]
    fn from(_: BytecodeCorruption) -> Self {
        crate::PyError::new(
            crate::PyErrorKind::BytecodeCorruption,
            "bytecode corruption",
        )
    }
}

/// CPython 3.14 `_PyCodeAddressRange`, ported through RustPython's code
/// object implementation.  This reads the authoritative `co_linetable`;
/// `CodeObject.locations` is an execution-oriented expansion which cannot
/// represent a missing line or column.
pub(crate) struct PyCodeAddressRange<'a> {
    pub(crate) ar_start: i32,
    pub(crate) ar_end: i32,
    pub(crate) ar_line: i32,
    computed_line: i32,
    reader: LineTableReader<'a>,
}

impl<'a> PyCodeAddressRange<'a> {
    pub(crate) fn new(linetable: &'a [u8], first_line: i32) -> Self {
        Self {
            ar_start: 0,
            ar_end: 0,
            ar_line: -1,
            computed_line: first_line,
            reader: LineTableReader::new(linetable),
        }
    }

    pub(crate) fn advance(&mut self) -> bool {
        let Some(first_byte) = self.reader.read_byte() else {
            return false;
        };
        // A byte in header position is decoded as one, bit 7 or not: the
        // marker separates a header from the payload bytes the skip below
        // consumes, and is not consulted here. `code.replace()` stores an
        // arbitrary `co_linetable`, and stopping on the marker made
        // `co_linetable=b"\0"` report no ranges where one entry is decoded.
        let code = (first_byte >> 3) & 0x0f;
        let length = ((first_byte & 0x07) + 1) as i32;
        let line_delta = get_line_delta(&self.reader, code);
        // Both accumulators wrap. `co_linetable` and `co_firstlineno` are
        // both writable from app-level, so a line delta chain can carry
        // `computed_line` past `i32` and a length chain can carry `ar_end`
        // there; upstream is built with `-fwrapv` and keeps walking, where a
        // plain `+` panics in a debug build.
        self.computed_line = self.computed_line.wrapping_add(line_delta);
        self.ar_line = if first_byte >> 3 == 0x1f {
            -1
        } else {
            self.computed_line
        };
        self.ar_start = self.ar_end;
        self.ar_end = self.ar_end.wrapping_add(length * 2);

        // Every payload byte has bit 7 clear; the next header has it set.
        while self.reader.peek_byte().is_some_and(|byte| byte & 0x80 == 0) {
            self.reader.read_byte();
        }
        true
    }

    /// `_PyCode_CheckLineNumber` — the line the byte offset `lasti` sits on,
    /// or `-1` when the walk runs out of ranges before reaching it.
    ///
    /// ```c
    /// while (bounds->ar_end <= lasti) {
    ///     if (!advance(bounds)) {
    ///         return -1;
    ///     }
    /// }
    /// while (bounds->ar_start > lasti) {
    ///     if (!retreat(bounds)) {
    ///         return -1;
    ///     }
    /// }
    /// return bounds->ar_line;
    /// ```
    ///
    /// The retreat loop has no counterpart here: a range built by
    /// [`PyCodeAddressRange::new`] starts before the first offset and this is
    /// the only walker, so `ar_start` never runs past `lasti`.
    ///
    /// `-1` covers two answers upstream also conflates: an offset past the
    /// last range, and an offset inside a `NO_LOCATION` range, which
    /// [`PyCodeAddressRange::advance`] reports by leaving `ar_line` at `-1`.
    fn check_line_number(&mut self, lasti: i32) -> i32 {
        while self.ar_end <= lasti {
            if !self.advance() {
                return -1;
            }
        }
        self.ar_line
    }

    /// The whole walk state except the table it reads.
    ///
    /// Upstream keeps `PyCodeAddressRange` inline in the two iterator objects,
    /// which own a reference to the code object.  A Rust borrow of
    /// `co_linetable` cannot outlive the call that took it, so the cursor
    /// travels between calls instead of the range, and the range is rebuilt on
    /// the same table each time.
    pub(crate) fn cursor(&self) -> LineTableCursor {
        LineTableCursor {
            pos: self.reader.pos,
            ar_start: self.ar_start,
            ar_end: self.ar_end,
            ar_line: self.ar_line,
            computed_line: self.computed_line,
        }
    }

    pub(crate) fn restore(&mut self, cursor: LineTableCursor) {
        self.reader.pos = cursor.pos;
        self.ar_start = cursor.ar_start;
        self.ar_end = cursor.ar_end;
        self.ar_line = cursor.ar_line;
        self.computed_line = cursor.computed_line;
    }

    pub(crate) fn resume(linetable: &'a [u8], cursor: LineTableCursor) -> Self {
        let mut range = Self::new(linetable, 0);
        range.restore(cursor);
        range
    }

    /// `advance_with_locations(bounds, endline, column, endcolumn)` — the
    /// walk `positionsiter_next` steps, which decodes the full location and
    /// not just the line.
    ///
    /// ```c
    /// int first_byte = *bounds->opaque.lo_next++;
    /// int code = (first_byte >> 3) & 15;
    /// bounds->ar_start = bounds->ar_end;
    /// bounds->ar_end = bounds->ar_start + ((first_byte & 7) + 1) * sizeof(_Py_CODEUNIT);
    /// switch(code) { ... }
    /// ```
    ///
    /// This one really does read the payload through the cursor and stops
    /// where the payload ends: there is no bit-7 scan, because upstream has
    /// none here.  `co_lines()` and `co_positions()` are therefore two
    /// separate walks that a hand-written `co_linetable` can make disagree,
    /// and both are reproduced rather than folded into one.
    ///
    /// The four values are returned instead of written through out-params.
    /// `-1` in any of them is the missing-offset marker
    /// [`source_offset_converter`] renders as `None`.
    fn advance_with_locations(&mut self) -> Option<(i32, i32, i32, i32)> {
        let first_byte = self.reader.read_byte()?;
        let code = (first_byte >> 3) & 0x0f;
        self.ar_start = self.ar_end;
        self.ar_end = self
            .ar_start
            .wrapping_add(((first_byte & 0x07) as i32 + 1) * 2);
        let (end_line, column, end_column) = match PyCodeLocationInfoKind::from_code(code) {
            Some(PyCodeLocationInfoKind::None) => {
                self.ar_line = -1;
                (-1, -1, -1)
            }
            Some(PyCodeLocationInfoKind::Long) => {
                self.computed_line = self
                    .computed_line
                    .wrapping_add(self.reader.read_signed_varint());
                self.ar_line = self.computed_line;
                let end_line = self.ar_line.wrapping_add(self.reader.read_varint() as i32);
                let column = (self.reader.read_varint() as i32).wrapping_sub(1);
                let end_column = (self.reader.read_varint() as i32).wrapping_sub(1);
                (end_line, column, end_column)
            }
            Some(PyCodeLocationInfoKind::NoColumns) => {
                self.computed_line = self
                    .computed_line
                    .wrapping_add(self.reader.read_signed_varint());
                self.ar_line = self.computed_line;
                (self.ar_line, -1, -1)
            }
            Some(
                kind @ (PyCodeLocationInfoKind::OneLine0
                | PyCodeLocationInfoKind::OneLine1
                | PyCodeLocationInfoKind::OneLine2),
            ) => {
                self.computed_line = self
                    .computed_line
                    .wrapping_add(kind.one_line_delta().unwrap_or(0));
                self.ar_line = self.computed_line;
                let column = self.reader.read_byte().unwrap_or(0) as i32;
                let end_column = self.reader.read_byte().unwrap_or(0) as i32;
                (self.ar_line, column, end_column)
            }
            // Short forms. `from_code` covers every value `(byte >> 3) & 15`
            // can take, so the `None` arm is unreachable and shares this one.
            _ => {
                let second_byte = self.reader.read_byte().unwrap_or(0);
                self.ar_line = self.computed_line;
                let column = ((code as i32) << 3) | ((second_byte >> 4) as i32);
                (self.ar_line, column, column + (second_byte & 0x0f) as i32)
            }
        };
        Some((self.ar_line, end_line, column, end_column))
    }
}

/// `get_line_delta(ptr)` — the entry's line delta, read **without moving**
/// the cursor.
///
/// ```c
/// static int
/// get_line_delta(const uint8_t *ptr)
/// {
///     int code = ((*ptr) >> 3) & 15;
///     switch (code) {
///         ...
///         case PY_CODE_LOCATION_INFO_LONG: return scan_signed_varint(ptr+1);
/// ```
///
/// `advance` moves solely by the byte scan that follows this call, stopping
/// at the next byte with bit 7 set. Consuming the payload here instead makes
/// the two disagree the moment a payload's declared varints run past that
/// byte — which `code.replace(co_linetable=...)` can store. `b"\x00\x80"`
/// is the shortest witness: the second byte is a `Short0` payload by length
/// and an entry header by its marker, and only the marker decides.
/// `co_lines()` reports one `(0, 4, ...)` range there, not `(0, 2, ...)`.
///
/// `reader` sits one byte past the header, where `ptr + 1` does, and `code`
/// is the caller's already-decoded `(*ptr >> 3) & 15`.
/// `_source_offset_converter(int *value)` — the marker `-1`, and only that
/// value, is the missing offset `co_positions()` reports as `None`.
///
/// ```c
/// static PyObject *
/// _source_offset_converter(int *value) {
///     if (*value == -1) {
///         Py_RETURN_NONE;
///     }
///     return PyLong_FromLong(*value);
/// }
/// ```
///
/// It runs on all four members of the tuple, so a line that *computes* to
/// `-1` is reported as `None` exactly like a `NO_LOCATION` range's is.
fn source_offset_converter(value: i32) -> PyObjectRef {
    if value == -1 {
        pyre_object::w_none()
    } else {
        w_int_new(value as i64)
    }
}

fn get_line_delta(reader: &LineTableReader<'_>, code: u8) -> i32 {
    let Some(kind) = PyCodeLocationInfoKind::from_code(code) else {
        return 0;
    };
    let mut reader = *reader;
    match kind {
        PyCodeLocationInfoKind::None => 0,
        PyCodeLocationInfoKind::Long => reader.read_signed_varint(),
        PyCodeLocationInfoKind::NoColumns => reader.read_signed_varint(),
        PyCodeLocationInfoKind::OneLine0 => 0,
        PyCodeLocationInfoKind::OneLine1 => 1,
        PyCodeLocationInfoKind::OneLine2 => 2,
        _ => 0,
    }
}

/// RustPython `LineTableReader`, matching CPython's 6-bit little-endian
/// location-table varints.
/// A suspended [`PyCodeAddressRange`] — every field of it but the borrowed
/// `co_linetable`.  See [`PyCodeAddressRange::cursor`].
#[derive(Clone, Copy)]
pub(crate) struct LineTableCursor {
    pos: usize,
    ar_start: i32,
    ar_end: i32,
    ar_line: i32,
    computed_line: i32,
}

#[derive(Clone, Copy)]
struct LineTableReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> LineTableReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_byte(&mut self) -> Option<u8> {
        let byte = self.data.get(self.pos).copied()?;
        self.pos += 1;
        Some(byte)
    }

    fn peek_byte(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Read one location-table varint: 6 bits per byte, **least
    /// significant group first**, bit 6 (0x40) marking continuation.
    ///
    /// This inverts `write_varint`, and deliberately differs from
    /// `decode_varint` below, which reads the exception table most
    /// significant group first to invert `parse_varint`. Both helpers sit
    /// a few lines apart in `pycore_code.h`: the byte order is a property
    /// of the table, not of the file. `_decode_varint` in PyPy's pycode.py
    /// is the exception-table reader despite its generic name, so reusing
    /// it here would decode every multi-byte line delta wrong.
    fn read_varint(&mut self) -> u32 {
        let Some(first) = self.read_byte() else {
            return 0;
        };
        let mut value = (first & 0x3f) as u32;
        let mut shift = 0;
        let mut byte = first;
        while byte & 0x40 != 0 {
            let Some(next) = self.read_byte() else {
                break;
            };
            shift += 6;
            // `code.replace(co_linetable=...)` stores arbitrary bytes, so the
            // continuation chain can run past what a u32 holds. Those groups
            // are unrepresentable either way; drop them rather than shift by
            // the full width, which panics in a debug build.
            if shift < u32::BITS {
                value |= ((next & 0x3f) as u32) << shift;
            }
            byte = next;
        }
        value
    }

    fn read_signed_varint(&mut self) -> i32 {
        let value = self.read_varint();
        if value & 1 != 0 {
            -((value >> 1) as i32)
        } else {
            (value >> 1) as i32
        }
    }
}

/// pycode.py _decode_varint — decode one CPython-3.11 varint at `i`.
///
/// Returns `(value, new_i)`. Reads 6 bits per byte, MSB first. Bit 6
/// (0x40) is the continuation flag; bit 7 (0x80) is the start-of-entry
/// marker, ignored here and masked off along with the continuation bit
/// via `& 63`.
///
/// Mirrors `parse_varint`. The location table uses the opposite byte
/// order — see `LineTableReader::read_varint` — so this decoder must not
/// be reused for `co_linetable`.
#[inline]
pub fn decode_varint(table: &[u8], mut i: usize) -> (u32, usize) {
    let mut b = table[i] as u32;
    i += 1;
    let mut value = b & 63;
    while b & 64 != 0 {
        b = table[i] as u32;
        i += 1;
        value = (value << 6) | (b & 63);
    }
    (value, i)
}

/// Decoded exception-table entry. Byte offsets throughout.
///
/// Field shape mirrors PyPy's `(start, length, target, depth, lasti)`
/// per-entry varint sequence; `end = start + length` is precomputed for
/// callers that want a half-open `start..end` range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExceptionTableEntry {
    pub start: u32,
    pub end: u32,
    pub target: u32,
    pub depth: u32,
    pub lasti: bool,
}

/// pycode.py `lookup_exceptiontable`.
///
/// Search `table` for a handler covering `instr_offset` (byte offset
/// into `co_code`). Returns `Some((target, depth, lasti))` when found,
/// `None` otherwise.
///
/// **Last matching wins**: entries are scanned in encoding order; if
/// multiple entries cover `instr_offset`, the later one (innermost in
/// CPython's emission order) is returned. Scanning short-circuits when
/// `start > instr_offset`, since entries are emitted in ascending
/// `start` order.
///
/// `pycode.py lookup_exceptiontable` is `@jit.elidable`: the table is
/// frozen on the code object, so a `(table, instr_offset)` pair folds.
#[majit_macros::elidable]
pub fn lookup_exceptiontable(table: &[u8], instr_offset: u32) -> Option<(u32, u32, bool)> {
    let n = table.len();
    if n == 0 {
        return None;
    }
    let mut best: Option<(u32, u32, bool)> = None;
    let mut i = 0;
    while i < n {
        let (start_raw, ni) = decode_varint(table, i);
        let start = start_raw * 2;
        let (length_raw, ni) = decode_varint(table, ni);
        let length = length_raw * 2;
        let (target_raw, ni) = decode_varint(table, ni);
        let target = target_raw * 2;
        let (dl, ni) = decode_varint(table, ni);
        let depth = dl >> 1;
        let lasti = (dl & 1) != 0;
        i = ni;
        if start <= instr_offset && instr_offset < start + length {
            best = Some((target, depth, lasti));
        } else if start > instr_offset {
            break;
        }
    }
    best
}

/// PyPy `astcompiler/codegen.py:2825-2826` / `generator.py:24-27`:
/// compute the `CO_YIELD_INSIDE_TRY` property once while constructing the
/// interpreter-level code object.
///
/// Python 3.14 wraps every generator body in a depth-zero, `lasti` exception
/// entry which only converts an escaping `StopIteration`; that synthetic
/// entry must not make every generator finalizable. Entries emitted for an
/// actual `try` around a yield either omit `lasti` at depth zero (`try`) or
/// carry a non-zero unwind depth (`with`). `lookup_exceptiontable` selects the
/// innermost entry, matching the compiler's `has_yield_inside_try` question.
fn code_yields_inside_try(code: &crate::CodeObject) -> bool {
    let mut index = 0;
    while index < code.instructions.len() {
        if matches!(
            code.instructions[index].op,
            crate::bytecode::Instruction::YieldValue { .. }
        ) {
            let offset = (index * 2) as u32;
            if let Some((_target, depth, lasti)) =
                lookup_exceptiontable(&code.exceptiontable, offset)
                && (depth != 0 || !lasti)
            {
                return true;
            }
        }
        index += 1;
    }
    false
}

/// Iterator over all decoded entries in `table`.
///
/// Convenience for callers that want a structural view (JIT codewriter,
/// liveness, the PyPy-style `mark_stacks` handler-shape seeder). The
/// runtime `handle_operation_error` dispatch uses [`lookup_exceptiontable`]
/// directly.
pub fn decode_exceptiontable(table: &[u8]) -> ExceptionTableIter<'_> {
    ExceptionTableIter { table, i: 0 }
}

pub struct ExceptionTableIter<'a> {
    table: &'a [u8],
    i: usize,
}

impl Iterator for ExceptionTableIter<'_> {
    type Item = ExceptionTableEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.table.len() {
            return None;
        }
        let (start_raw, i) = decode_varint(self.table, self.i);
        let start = start_raw * 2;
        let (length_raw, i) = decode_varint(self.table, i);
        let length = length_raw * 2;
        let (target_raw, i) = decode_varint(self.table, i);
        let target = target_raw * 2;
        let (dl, i) = decode_varint(self.table, i);
        self.i = i;
        Some(ExceptionTableEntry {
            start,
            end: start + length,
            target,
            depth: dl >> 1,
            lasti: (dl & 1) != 0,
        })
    }
}

/// Compatibility container for code-hook caching state.
#[derive(Debug, Default)]
pub struct CodeHookCache {
    _code_hook: Option<PyObjectRef>,
}

/// Type descriptor for code objects.
pub static CODE_TYPE: PyType =
    pyre_object::pyobject::new_pytype_with_weakref("code", CODE_W_WEAKREFLIFELINE_OFFSET);

/// `pycode.py` `PyCode._immutable_fields_` entries that live on the
/// `bytecode::CodeObject` body [`PyCode::code_ptr`] addresses:
/// `co_varnames[*]`, `co_cellvars[*]`, `co_freevars[*]`.
/// `localspluskinds[*]` is the kind byte of that layout
/// (`CO_FAST_LOCAL` / `CO_FAST_CELL` / `CO_FAST_FREE`).
/// `code_replace` writes the clone; `release_code_object_body` clears a
/// body only after the graph is unpublished. The harvester keys this
/// const's bare leaf `CodeObject`, which is the field read's `owner_root`.
/// `source_path` stays mutable (`fix_code_filenames`).
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _immutable_fields_CodeObject: &str =
    "varnames[*],cellvars[*],freevars[*],localspluskinds[*]";

/// Host `bytecode::CodeObject` is `Box::into_raw`'d host memory, never a
/// GC object. `rclass.py getgcflavor` reads `_alloc_flavor_` off the
/// classdesc so `getinstancerepr` builds a raw `InstanceRepr`.
#[doc(hidden)]
#[allow(non_upper_case_globals, dead_code)]
pub const _alloc_flavor_CodeObject: &str = "raw";

/// Python code object wrapper.
///
/// Stores a raw pointer to the bytecode `CodeObject`. A top-level body is
/// `Box::into_raw`'d; a nested body points into that permanently-live owner,
/// matching PyPy's one recursively-owned `co_consts_w` code graph without
/// cloning the child graph at each wrapping boundary.
///
/// `w_globals?` is `pycode.py` `PyCode._immutable_fields_`: the first store
/// fills it, and a later read of the promoted code object is quasi-immutable.
/// `code_ptr` is the permanently-live body `w_code_new_owned` stores before
/// the wrapper is published and never writes again. `PyFrame.getcode`
/// promotes `self.pycode` and then reads that pointer; listing it here makes
/// the read a constant, the same way `co_code` and `co_varnames` are
/// immutable on `PyCode`. The body's name tables and kind bytes are
/// [`_immutable_fields_CodeObject`]. `source_path` stays mutable
/// (`fix_code_filenames`).
#[repr(C)]
#[majit_macros::jit_immutable_fields(
    "w_globals?",
    "code_ptr",
    "co_consts_w[*]",
    "co_names_w[*]",
    "co_nlocals",
    "co_stacksize",
    "co_nfreevars",
    "co_flags"
)]
pub struct PyCode {
    pub ob_header: PyObject,
    /// Permanently-live `CodeObject` address. Top-level bodies are owned via
    /// `Box::into_raw`; nested bodies borrow from one of those boxes. The
    /// pointee is a host allocation (`history.getkind` of a `Ptr` whose
    /// `TO._gckind == 'raw'` is `"int"`), not a GC object.
    pub code_ptr: *const crate::CodeObject,
    /// `pycode.py self.co_firstlineno = firstlineno`. RustPython's
    /// `CodeObject.first_line_number: Option<OneIndexed>` cannot represent
    /// the zero/negative values accepted by Python 3.14's CodeType
    /// constructor, so preserve the exact Python integer on the PyCode itself.
    pub co_firstlineno_raw: i32,
    /// `pycode.py self.co_filename = filename`: the byte-exact filesystem
    /// spelling owned by this `PyCode`. Null means that `code.source_path` is
    /// already the exact UTF-8 spelling, avoiding an allocation for compiled
    /// source paths and ordinary filename replacements.
    ///
    /// [`PyCode::w_filename`] is the `pycode.py PyCode.__init__`
    /// `space.newfilename(filename)` object, realized before the wrapper is
    /// published and refreshed by [`set_filename_bytes`]
    /// (`importing.py update_code_filenames`). That string is a movable
    /// managed object, so `eval::walk_enrolled_code_roots` traces and
    /// forwards the slot the same way it forwards `w_name`.
    pub filename_bytes: *mut Vec<u8>,
    /// PyPy `pycode.py:132 self.co_code = co_code`: byte-exact public
    /// bytecode supplied to `CodeType` / `code.replace` when it contains an
    /// opcode that RustPython's enum cannot represent (or its explicit
    /// `Reserved` hole).  Ordinary compiler-produced code leaves this null
    /// and derives `co_code` from `CodeUnits`.
    ///
    /// The execution stream uses `Instruction::Reserved` as a placeholder,
    /// but getters, equality, hashing and marshal must retain the actual byte.
    /// Keeping the owner-local field mirrors PyPy's `PyCode.co_code`; an
    /// address-keyed side table would lose both lifetime and structural parity.
    pub co_code_bytes: *mut Vec<u8>,
    /// Whether a nested compiler constant selected by
    /// `importing.py update_code_filenames`' `oldname` guard inherits
    /// `filename_bytes` if a fallback slot ever has to be rebuilt. False for
    /// `pycode.py` constructor/replace filenames, which affect only the
    /// code object being constructed.
    pub filename_inherits_to_nested: bool,
    /// PyPy: `PyCode.w_globals` — the globals dict OBJECT (`W_DictMultiObject`,
    /// `pycode.py "w_globals?"`).  Module globals are `malloc_typed`-
    /// immortal, but `exec`/custom-globals dicts are `try_gc_alloc` movable.
    /// Managed wrappers trace this slot through `eval::walk_raw_code_roots`.
    /// Bootstrap wrappers outside the collector are registered in
    /// [`W_GLOBALS_STAMPED_CODES`] and forwarded from there. Null until first
    /// stamped by `frame_stores_global`.
    pub w_globals: PyObjectRef,
    /// `typedef.py make_weakref_descr(PyCode)` installs `_lifeline_` on
    /// the interpreter-level class.  Keep that owner-local field here rather
    /// than routing code objects through the fallback address-keyed table.
    /// The lifeline owns the cached weakrefs and their callbacks and is traced
    /// with the rest of this code object's managed children.
    pub w_weakreflifeline: PyObjectRef,
    /// PyPy: `PyCode.hidden_applevel` (`pycode.py, 147`). Set by
    /// `pycompiler.compile(hidden_applevel=True)` for the app-level half of a
    /// mixed module (`gateway.py ApplevelClass`), and by
    /// `interp_magic.py hidden_applevel(func)` for one function at a time.
    /// `pyframe.py hide` reads it, which is what keeps such a frame out of the
    /// trace hook (`executioncontext.py _trace`), out of a recorded traceback
    /// (`pytraceback.py record_application_traceback`) and out of the
    /// `getnextframe_nohidden` walk `sys._getframe` counts along.
    pub hidden_applevel: bool,
    /// pycode.py `_compute_flatcall`. Cached arity descriptor:
    /// - 0-4: impossible (builtins only)
    /// - FLATPYCALL | co_argcount: simple user function
    /// - HOPELESS: has *args/**kwargs/kwonly/too many params
    /// The unused high bit caches `CO_YIELD_INSIDE_TRY`; accessors mask it
    /// away from the arity value.
    pub fast_natural_arity: u16,
    /// Cached [`crate::pyframe::npure_cellvars`] — the count of cellvars that
    /// are not also varnames.  Code-invariant, so computed once here instead
    /// of re-walking the O(cellvars × varnames) overlap check on every
    /// `PyFrame::ncells()` / stack-base query (a per-`pop_value` hot path).
    /// `u32::MAX` sentinel when `code_ptr` is null/unaligned (test stubs).
    pub npure_cellvars: u32,
    /// `pycode.py self.co_nlocals = nlocals` (`_immutable_fields_ co_nlocals`)
    /// — the `varnames` count.
    pub co_nlocals: u32,
    /// `pycode.py self.co_stacksize = stacksize` (`_immutable_fields_
    /// co_stacksize`).
    pub co_stacksize: u32,
    /// `len(pycode.py self.co_freevars)`. The array itself is
    /// `_immutable_fields_ co_freevars[*]`; this length is the same fact.
    pub co_nfreevars: u32,
    /// `pycode.py self.co_flags = flags` (`_immutable_fields_ co_flags`).
    pub co_flags: u32,
    /// `pycode.py` `PyCode._initialize`, `self.cell_families = [CellFamily(name) for name in
    /// cellvars]` — one [`CellFamily`] per cellvar, shared by every cell any
    /// frame of this code creates for that cellvar.
    ///
    /// Indexed by localsplus slot rather than by cellvar position: pyre
    /// follows the 3.11 unified slot layout, where a cellvar that also names a
    /// parameter shares that parameter's slot and the rest occupy the
    /// pure-cellvar band, so both creation sites
    /// ([`crate::pyframe::PyFrame::initialize_frame_scopes`] and `MAKE_CELL`)
    /// index in O(1).  Slots naming no cellvar hold null; `null` table when
    /// `code_ptr` is invalid or the code has no cellvars (pycode.py `PyCode._initialize`).
    ///
    /// The `Vec` is owned via `Box::into_raw` and dropped with the wrapper, but
    /// the families it points at are leaked: a cell outlives the code object
    /// that made it whenever a closure survives its enclosing function, and
    /// `Cell.family` is a raw, untraced pointer.
    pub cell_families: *mut Vec<*const CellFamily>,
    /// `pycode.py self._globals_caches = [None] * len(self.co_names_w)`.
    ///
    /// Per-name slot for `LOAD_GLOBAL_cached` / `STORE_GLOBAL_cached`
    /// (`celldict.py`).  Stores a weak reference to
    /// the `GlobalCache` resolved on the first miss, so subsequent
    /// hits bypass the `mstrategy.get_global_cache(varname)` string
    /// lookup.
    ///
    /// Owned via `Box::into_raw`; allocated once at construction sized
    /// to `code.names.len()`, never resized.  `null` when `code_ptr`
    /// is null or unaligned (test fixtures, gateway builtins).
    pub globals_caches: *mut parking_lot::Mutex<
        Vec<Option<std::sync::Weak<parking_lot::Mutex<pyre_object::celldict::GlobalCache>>>>,
    >,
    /// `mapdict.py self._mapdict_caches = [INVALID_CACHE_ENTRY] *
    /// len(co_names_w)`.
    ///
    /// Per-name slot for the `LOAD_ATTR_caching` / `STORE_ATTR_caching` inline
    /// attribute cache (`mapdict.py/1574`).  A `None` slot is PyPy's
    /// `INVALID_CACHE_ENTRY` (mapdict.py); a `Some` holds the immortal map
    /// node + attribute node + `version_tag` last resolved for this slot, so a
    /// monomorphic re-read skips the type lookup + map walk.  The
    /// LOAD_METHOD fill additionally stores a movable `w_method`
    /// reference (mapdict.py), forwarded during collection by
    /// `walk_mapdict_method_cache_gc`; the other fields are immortal
    /// node pointers and need no walking.
    ///
    /// Owned via `Box::into_raw`, sized to `code.names.len()` at construction,
    /// never resized; `null` when `code_ptr` is null or unaligned.
    pub mapdict_caches: *mut Vec<Option<crate::objspace::std::mapdict::MapdictCacheEntry>>,
    /// `pycode.py self.co_consts_w = consts` (`_immutable_fields_
    /// co_consts_w[*]`). `getconstant_w(index)` (`pyopcode.py`) is
    /// `co_consts_w[index]`.
    ///
    /// PyPy receives this list already wrapped from the compiler. Pyre's
    /// compiler keeps `ConstantData` unwrapped, so the `PyCode` constructor
    /// wraps every slot before publishing the code object. This is observable:
    /// `gc.get_objects()` must not gain a permanent object the first time a
    /// `LOAD_CONST` executes.
    ///
    /// The table is a type-9 `FixedObjectArray` (`Ptr(GcArray(OBJECTPTR))`),
    /// the host shape of the `co_consts_w` list. `FixedObjectArray::set_ref` is the
    /// `setarrayitem_gc` rewrite: it write-barriers the array so a minor
    /// that remembers it copy+scans the items. A code object whose `code_ptr` is
    /// null or unaligned (test fixtures, gateway builtins) holds an empty
    /// array, so `getconstant_w` is the bounds-checked array read.
    pub co_consts_w: *mut FixedObjectArray,
    /// `pycode.py self.co_names_w = [space.new_interned_str(aname) for aname
    /// in names]` (`_immutable_fields_ co_names_w[*]`). `getname_w(index)`
    /// (`pyopcode.py`) is `co_names_w[index]`.
    ///
    /// `PyCode.__init__` interns the whole list before the code object is
    /// published (`new_interned_str`). `getname_w` still compare-exchanges a
    /// null slot, for a table built before this fill.
    ///
    /// Slots hold `intern_str_value` / `new_interned_w_str` results. The table
    /// is a type-9 `FixedObjectArray` (`Ptr(GcArray(OBJECTPTR))`), the same
    /// allocation family as `co_consts_w`, so a minor copies the array and
    /// scans the items. First publication is an atomic compare-exchange on
    /// the item word: free-threaded readers share one object, and `set_ref`
    /// write-barriers the array after the winner stores. A `null` slot is
    /// unrealized. A code object whose `code_ptr` is null or unaligned, or
    /// whose `co_names` is empty, holds a zero-length array — never null
    /// (`pycode.py PyCode.__init__` builds `[]` when `names` is empty).
    pub co_names_w: *mut FixedObjectArray,
    /// `pycode.py self.co_qualname = qualname` realized as one shared
    /// wrapped object.
    ///
    /// `function.py self.qualname = qualname or self.name` copies the code
    /// object's interp-level string into every function built from it, so all
    /// of them name the same immutable string. Pyre wraps names as objects, so
    /// the code object owns the single realized instance and both
    /// `MAKE_FUNCTION` and the `co_qualname` getter hand back that object
    /// instead of allocating a fresh `W_UnicodeObject` per read.
    ///
    /// `PY_NULL` until first realized by [`w_code_qualname_obj`], which builds
    /// it with `w_str_new` — `malloc_typed`-immortal, so the stored pointer is
    /// fixed.  `eval::walk_raw_code_roots` forwards the slot anyway, for the
    /// same reason it forwards `w_globals`: the walker serves both the managed
    /// custom trace and the bootstrap prebuilt-root walk.
    pub w_qualname: PyObjectRef,
    /// `pycode.py self.co_name = name` realized as one shared wrapped
    /// object, the exact counterpart of `w_qualname` above.
    ///
    /// `function.py self.name = forcename or code.co_name` copies the code
    /// object's interp-level string into every function built from it, so the
    /// same "all of them name one immutable string" argument applies, and the
    /// same realize-once treatment follows.  Without it the getter ran
    /// `w_str_new` per read — two allocations, a `String` copy and a full
    /// `chars().count()` — which a traceback walk pays on every hop
    /// (`tb.tb_frame.f_code.co_name`).
    ///
    /// `PY_NULL` until first realized by [`w_code_name_obj`]; the storage,
    /// immortality and root-walk argument are `w_qualname`'s verbatim.
    pub w_name: PyObjectRef,
    /// `pycode.py self.w_filename = space.newfilename(filename)` realized as
    /// one shared wrapped object.
    ///
    /// `typedef.py co_filename = interp_attrproperty_w('w_filename', cls=PyCode)`
    /// hands this slot back on every read, so two `co_filename` lookups name
    /// one object. `PyCode.__init__` stores it before the code object is
    /// published (`_immutable_fields_` lists `w_filename`);
    /// `importing.py update_code_filenames` is the later writer. The realized
    /// string is a movable managed string, so the slot is traced and forwarded
    /// by `eval::walk_enrolled_code_roots`.
    pub w_filename: PyObjectRef,
    /// The last `(addrq, line)` pair [`w_code_addr2line`] resolved for this
    /// code object, packed as `addrq << 32 | line as u32`.
    ///
    /// `_PyCode_CheckLineNumber` restarts the linetable decode at the first
    /// range on every call, so an offset deep in a function costs a walk
    /// proportional to how far in it sits.  The offsets asked about repeat:
    /// `pytraceback.py record_application_traceback` resolves `tb_lineno` at
    /// the moment the traceback node is built, so a line that raises in a loop
    /// asks the same question every iteration, and `f_lineno` on a suspended
    /// frame does the same.  Both inputs the walk reads — `co_linetable` on
    /// the `CodeObject` and [`PyCode::co_firstlineno_raw`] — are fixed once
    /// the wrapper is published (`code.replace` boxes a fresh one), so a
    /// repeat of the previous question already has its answer.
    ///
    /// [`ADDR2LINE_MEMO_EMPTY`] is the unset value: it spells an `addrq` no
    /// memoised query carries, since the negative offsets answer
    /// `co_firstlineno` without walking anything.
    ///
    /// Direct-mapped over [`ADDR2LINE_MEMO_WAYS`] slots rather than a single
    /// one: the offsets a loop asks about alternate (the raising instruction
    /// and the handler that reads `tb_lineno` sit on different lines of the
    /// same code object), and one slot answers only whichever of them asked
    /// last.
    pub addr2line_memo: [std::sync::atomic::AtomicI64; ADDR2LINE_MEMO_WAYS],
    /// Per-code loop-header set.
    ///
    /// `pycode.py` `PyCode._initialize` keeps derived per-code data on the
    /// code object (`_args_as_cellvars`, `cell_families`, `_globals_caches`).
    /// The loop-header set has no upstream counterpart: `interp_jit.py`
    /// `jump_absolute` only compares `jumpto >= next_instr`. pyre scans
    /// bytecode because 3.14 emits handler-rejoin backward jumps, so the
    /// scan lives here (`LoopHeaderInfo`) instead of an address-keyed side
    /// table.
    ///
    /// Null until the first query. Built from `code_ptr` and published with
    /// `compare_exchange`; the loser drops its box. Stays null when
    /// `code_ptr` is null or unaligned, the same case `globals_caches` leaves
    /// empty.
    pub loop_header_info: std::sync::atomic::AtomicPtr<crate::loop_headers::LoopHeaderInfo>,
    /// The hidden `mutate_w_globals` of `pycode.py` `"w_globals?"`.
    ///
    /// Holds no GC pointers. The wrapper is born non-moving
    /// (`malloc_typed_stable`), which is what lets a recorded watcher keep
    /// this address. The collector runs no `Drop`, so [`pycode_destructor`]
    /// reclaims the instance.
    pub w_globals_watchers: pyre_object::quasiimmut::QuasiImmutField,
    /// Per-code result of the runtime frame-shape gate.
    ///
    /// 0 is uncomputed; a later fill stores `1 +` the shape discriminant.
    /// `pycode.py` keeps per-code JIT state on `PyCode` (`jit_cells`); this
    /// is the same owner for the pyre-only shape latch.
    pub jit_shape: std::sync::atomic::AtomicU8,
}

/// Field offset of `code_ptr` within `PyCode`.
pub const CODE_PTR_OFFSET: usize = std::mem::offset_of!(PyCode, code_ptr);
/// Field offset of `w_globals` within `PyCode`.
pub const CODE_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(PyCode, w_globals);
/// Field offset of `w_weakreflifeline` within `PyCode`.
pub const CODE_W_WEAKREFLIFELINE_OFFSET: usize = std::mem::offset_of!(PyCode, w_weakreflifeline);
/// Field offset of `w_qualname` within `PyCode`.
pub const CODE_W_QUALNAME_OFFSET: usize = std::mem::offset_of!(PyCode, w_qualname);
/// Field offset of `w_name` within `PyCode`.
pub const CODE_W_NAME_OFFSET: usize = std::mem::offset_of!(PyCode, w_name);
/// Field offset of `w_filename` within `PyCode`.
pub const CODE_W_FILENAME_OFFSET: usize = std::mem::offset_of!(PyCode, w_filename);
/// Slots in [`PyCode::addr2line_memo`].  A power of two: the slot is picked
/// by masking the offset.
pub const ADDR2LINE_MEMO_WAYS: usize = 4;

/// [`PyCode::addr2line_memo`] before any offset has been resolved.
///
/// `addrq` is `i32::MIN`, which the memoised arm never stores: it takes only
/// non-negative offsets.
pub const ADDR2LINE_MEMO_EMPTY: i64 = i64::MIN;

/// Field offset of `co_firstlineno_raw` within `PyCode`.
pub const CODE_CO_FIRSTLINENO_RAW_OFFSET: usize = std::mem::offset_of!(PyCode, co_firstlineno_raw);
/// Field offset of `hidden_applevel` within `PyCode`.
pub const CODE_HIDDEN_APPLEVEL_OFFSET: usize = std::mem::offset_of!(PyCode, hidden_applevel);

/// `PyCode.jit_shape` before the first classification.
pub const JIT_SHAPE_UNCOMPUTED: u8 = 0;
/// Stored `jit_shape` for assembler overflow: `1 + ConstEncodingOverflow`.
pub const JIT_SHAPE_STORED_CONST_ENCODING_OVERFLOW: u8 = 4;

/// The `co_firstlineno` slot, exactly as [`code_get_field`] reads it.
///
/// # Safety
/// `w_code` must be a live `PyCode`.
pub unsafe fn w_code_firstlineno_raw(w_code: PyObjectRef) -> i32 {
    unsafe { (*(w_code as *const PyCode)).co_firstlineno_raw }
}

/// `pycode.py self.co_qualname = qualname` — the shared wrapped qualified
/// name, realized on first demand and retained on the code object.
///
/// `function.py self.qualname = qualname or self.name` and the `co_qualname`
/// getter both hand out the code object's own string, so they name one
/// immutable value; realizing it once here reproduces that identity instead of
/// minting a `W_UnicodeObject` per `def` and per attribute read.  Returns
/// `PY_NULL` for a wrapper with no code body (test stubs, gateway builtins),
/// whose callers already handle the missing name.
///
/// # Safety
/// `w_code` must point to a valid `PyCode`.
pub unsafe fn w_code_qualname_obj(w_code: PyObjectRef) -> PyObjectRef {
    unsafe {
        let roots = pyre_object::gc_roots::push_roots();
        let code_slot = roots.base();
        let w_code = roots.pin_root(w_code);
        let cached = (*(w_code as *const PyCode)).w_qualname;
        if !cached.is_null() {
            return cached;
        }
        let code_ptr = w_code_get_ptr(w_code) as *const crate::CodeObject;
        if code_ptr.is_null() {
            return pyre_object::PY_NULL;
        }
        // `w_str_new` is a collection point. RPython keeps the live `self`
        // reference in the shadow stack across it; reload the possibly moved
        // wrapper before publishing the cache field.
        let w_qualname = w_str_new(&(*code_ptr).qualname);
        let published = publish_code_slot_store_rooting(roots.get(code_slot), &[w_qualname]);
        let w_qualname = published.get(0);
        (*(published.owner() as *mut PyCode)).w_qualname = w_qualname;
        w_qualname
    }
}

/// `pycode.py self.co_name = name` — the shared wrapped name, realized on
/// first demand and retained on the code object.
///
/// The counterpart of [`w_code_qualname_obj`], for the same reason and with the
/// same lifetime argument: `function.py:51 self.name = forcename or
/// code.co_name` and the `co_name` getter both hand out the code object's own
/// string, so they name one immutable value.
///
/// # Safety
/// `w_code` must point to a valid `PyCode`.
pub unsafe fn w_code_name_obj(w_code: PyObjectRef) -> PyObjectRef {
    unsafe {
        let roots = pyre_object::gc_roots::push_roots();
        let code_slot = roots.base();
        let w_code = roots.pin_root(w_code);
        let cached = (*(w_code as *const PyCode)).w_name;
        if !cached.is_null() {
            return cached;
        }
        let code_ptr = w_code_get_ptr(w_code) as *const crate::CodeObject;
        if code_ptr.is_null() {
            return pyre_object::PY_NULL;
        }
        // `w_str_new` is a collection point. RPython keeps the live `self`
        // reference in the shadow stack across it; reload the possibly moved
        // wrapper before publishing the cache field.
        let w_name = w_str_new(&(*code_ptr).obj_name);
        let published = publish_code_slot_store_rooting(roots.get(code_slot), &[w_name]);
        let w_name = published.get(0);
        (*(published.owner() as *mut PyCode)).w_name = w_name;
        w_name
    }
}

/// `objspace.py newfilename` as `pycode.py PyCode.__init__` and
/// `importing.py update_code_filenames` use it: wrap the code object's
/// current filename spelling. `filename_bytes` is the filesystem form when
/// it is not UTF-8; otherwise `code.source_path` is already the exact
/// spelling.
///
/// # Safety
/// When `filename_bytes` is null, `code_ptr` must point at a live
/// `CodeObject`. `filename_bytes`, when non-null, must point at the
/// `PyCode` field's `Vec`.
unsafe fn realize_w_filename(
    filename_bytes: *const Vec<u8>,
    code_ptr: *const crate::CodeObject,
) -> PyObjectRef {
    if filename_bytes.is_null() {
        w_str_new_managed(&unsafe { &*code_ptr }.source_path)
    } else {
        crate::gateway::fsdecode_filename_bytes(unsafe { &*filename_bytes })
    }
}

/// `pycode.py self.w_filename = space.newfilename(filename)` — the shared
/// wrapped filename stored by `PyCode.__init__` and refreshed by
/// `importing.py update_code_filenames`.
///
/// `typedef.py co_filename = interp_attrproperty_w('w_filename', cls=PyCode)`
/// and every other reader hand out the code object's own string, so they
/// name one immutable value. The filesystem decode (`objspace.py newfilename`)
/// returns a movable managed string, which `eval::walk_enrolled_code_roots`
/// traces and forwards.
///
/// # Safety
/// `w_code` must point to a valid `PyCode` whose `w_filename` was realized
/// at construction.
pub unsafe fn w_code_filename_obj(w_code: PyObjectRef) -> PyObjectRef {
    let w_filename = unsafe { (*(w_code as *const PyCode)).w_filename };
    debug_assert!(
        !w_filename.is_null(),
        "PyCode.w_filename is realized at construction"
    );
    w_filename
}

/// GC type id assigned to `PyCode`.
///
/// `PyCode` is a normal interpreter-level code object in PyPy
/// (`pycode.py class PyCode(eval.Code)`).  This tid is pinned by
/// a `debug_assert_eq!` in the pyre-jit type-registration sequence: the
/// `PyCode` `TypeInfo` is registered explicitly just before the
/// foreign-pytype loop, taking the slot directly after
/// the two `GcArray` tids (41 and 42 in that order).  Pre-registering it there (and
/// inserting `CODE_TYPE` into `pytype_to_tid`) makes the foreign loop
/// skip `CODE_TYPE`, so the net register-call count up to
/// `W_MODULE_DICT_GC_TYPE_ID = 48` is unchanged and no downstream tid
/// shifts.  The numeric value coincides with the dormant
/// `pytraceback::PYTRACEBACK_GC_TYPE_ID` constant, but `PyTraceback`
/// is still host-allocated and is never GC-registered, so tid 43 only
/// ever tags a `PyCode` at runtime and the two do not collide.
pub const W_CODE_GC_TYPE_ID: u32 = 43;

impl pyre_object::lltype::GcType for PyCode {
    fn type_id() -> u32 {
        W_CODE_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<PyCode>();
}

/// Compatibility helper for unpacking a tuple of strings.
pub fn unpack_text_tuple(_space: PyObjectRef, w_str_tuple: PyObjectRef) -> Vec<String> {
    let _ = (_space, w_str_tuple);
    Vec::new()
}

/// Compatibility API for building a signature-like object.
pub fn make_signature(_code: &PyCode) -> PyObjectRef {
    let _ = _code;
    pyre_object::w_none()
}

/// pycode.py _compute_args_as_cellvars
pub fn _compute_args_as_cellvars(
    varnames: &[String],
    cellvars: &[String],
    argcount: usize,
) -> Vec<isize> {
    let mut args_as_cellvars = Vec::new();
    for i in 0..cellvars.len() {
        let cellname = &cellvars[i];
        for j in 0..argcount {
            if *cellname == varnames[j] {
                while args_as_cellvars.len() < i {
                    args_as_cellvars.push(-1isize);
                }
                args_as_cellvars.push(j as isize);
            }
        }
    }
    args_as_cellvars
}

#[inline]
pub fn _code_const_eq(_space: PyObjectRef, w_a: PyObjectRef, w_b: PyObjectRef) -> bool {
    let _ = _space;
    unsafe { code_const_equal(w_a, w_b).unwrap_or(false) }
}

pub fn _convert_const(_space: PyObjectRef, w_a: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    // PyPy `pycode._convert_const`, including the recursive tuple/frozenset
    // construction.  Keep the input in a shadow-stack slot because every
    // converted child and result container may collect.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(w_a);
    let current = || pyre_object::gc_roots::shadow_stack_get(root_base);
    let w_a = current();

    let is_str = unsafe { is_exact_type(w_a, &STR_TYPE) };
    let is_bytes = unsafe { is_exact_type(w_a, &pyre_object::bytesobject::BYTES_TYPE) };
    if is_str || is_bytes {
        // `_convert_const`: `(w_type, w_a)` keeps text and bytes separated
        // while comparing each payload by value.
        let w_type = crate::typedef::r#type(current())
            .map(std::ptr::NonNull::as_ptr)
            .unwrap_or(std::ptr::null_mut());
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_type);
        fields.push(current());
        return w_tuple_new(fields.take());
    }
    if unsafe { is_code(w_a) } {
        return current();
    }
    if unsafe { is_exact_type(w_a, &TUPLE_TYPE) } {
        let len = unsafe { pyre_object::w_tuple_len(w_a) };
        let mut items = pyre_object::gc_roots::RootedItems::new();
        for index in 0..len {
            let item = unsafe { pyre_object::w_tuple_getitem(current(), index as i64) }
                .unwrap_or(pyre_object::w_none());
            items.push(_convert_const(std::ptr::null_mut(), item));
        }
        return w_tuple_new(items.take());
    }
    if unsafe { is_exact_type(w_a, &pyre_object::setobject::FROZENSET_TYPE) } {
        // `w_set_len` counts live elements while `w_set_key_at` is keyed on a
        // storage slot, and `w_set_copy_storage_from` clones the storage
        // whole, tombstones included.  Walk the cursor, as every other set
        // walk does.
        let mut items = pyre_object::gc_roots::RootedItems::new();
        let mut cursor = 0;
        while let Some(slot) = unsafe { pyre_object::setobject::w_set_next_slot(current(), cursor) }
        {
            if let Some(key) = unsafe { pyre_object::setobject::w_set_key_at(current(), slot) } {
                items.push(_convert_const(std::ptr::null_mut(), key.obj));
            }
            cursor = slot + 1;
        }
        return pyre_object::w_frozenset_from_items(&items.take());
    }
    if unsafe { pyre_object::sliceobject::is_slice(w_a) } {
        // [3.14-spec] PyPy 3.11 has no marshal-v5 slice constant.  Preserve
        // its recursive-key shape for the new public constant kind.
        let w_type = crate::typedef::r#type(current())
            .map(std::ptr::NonNull::as_ptr)
            .unwrap_or(std::ptr::null_mut());
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_type);
        // Each conversion allocates, so read the next field off the rooted
        // slice rather than snapshotting all three up front.
        fields.push(_convert_const(std::ptr::null_mut(), unsafe {
            pyre_object::sliceobject::w_slice_get_start(current())
        }));
        fields.push(_convert_const(std::ptr::null_mut(), unsafe {
            pyre_object::sliceobject::w_slice_get_stop(current())
        }));
        fields.push(_convert_const(std::ptr::null_mut(), unsafe {
            pyre_object::sliceobject::w_slice_get_step(current())
        }));
        return w_tuple_new(fields.take());
    }
    if unsafe { is_exact_type(w_a, &COMPLEX_TYPE) } {
        let real = unsafe { pyre_object::w_complex_get_real(w_a) };
        let imag = unsafe { pyre_object::w_complex_get_imag(w_a) };
        if !real.is_nan() && !imag.is_nan() {
            // PyPy's complex immutable id carries the component bit patterns.
            // Public `id(complex)` follows 3.14's address identity in Pyre, so
            // spell that private strong key here instead of changing `id()`.
            let w_type = crate::typedef::r#type(current())
                .map(std::ptr::NonNull::as_ptr)
                .unwrap_or(std::ptr::null_mut());
            let mut fields = pyre_object::gc_roots::RootedItems::new();
            fields.push(w_type);
            fields.push(w_int_new(real.to_bits() as i64));
            fields.push(w_int_new(imag.to_bits() as i64));
            return w_tuple_new(fields.take());
        }
    }
    crate::baseobjspace::id(current())
}

/// pypy/interpreter/pycode.py `PyCode.__init__`
/// (`hidden_applevel` field assignment, line 147).
///
/// ```python
/// def __init__(self, space, ..., hidden_applevel=False, magic=default_magic):
///     ...
///     self.hidden_applevel = hidden_applevel
/// ```
///
/// `w_code_new(code_ptr)` is the `hidden_applevel=False` default
/// shorthand; callers who need the flag set (mirroring PyPy's
/// `BuiltinCode` (gateway.py) / `ApplevelClass`
/// (gateway.py) / `_continuation` entrypoint dummy
/// (interp_continuation.py)) construct via this entry point.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a permanently-live `CodeObject`,
/// either obtained via `Box::into_raw` or nested inside one such owner.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the body
/// boxes the `PyCode` through the managed allocator and its per-name cache
/// tables through direct raw allocations. Residualise the whole
/// constructor — a `PyObjectRef` GCREF modelled by signature. Code objects are
/// built at import/compile time, never on a traced hot path.
///
/// `pycode.py` makes `PyCode` an ordinary GC object. Pyre's current JIT seam,
/// however, still carries raw wrapper addresses alongside `CodeObject*` in
/// compiled and resume metadata. Keep the wrapper collector-owned but
/// stable-address until those remaining raw couriers are replaced with GC
/// references; the collector can still reclaim it and its namespace cycle.
/// Bootstrap wrappers created before GC installation fall back to the
/// prebuilt family and use the explicit root walkers.
#[majit_macros::dont_look_inside]
pub fn w_code_new_with_hidden_applevel(code_ptr: *const (), hidden_applevel: bool) -> PyObjectRef {
    // Zero owner: this entry point's contract is that the body is never
    // released, so the wrapper takes no part in `CodeUnit` retirement.
    w_code_new_owned(code_ptr, hidden_applevel, 0, &[], true)
}

/// [`w_code_new_with_hidden_applevel`] naming the `box_code_object` allocation
/// the body belongs to, so the wrapper can be counted against it.
///
/// `pycode.py self.co_consts_w = consts` as a type-9 `FixedObjectArray`.
///
/// The `co_consts_w` list is a young GC object; a minor that reaches the `PyCode`
/// copies the list and then scans the items. pyre births `PyCode`
/// old-gen, so the table is allocated in the nursery — `co_consts_w` is
/// then an old-to-young edge whose write barrier remembers the wrapper,
/// and `walk_raw_code_roots` visiting the array pointer is the copy+scan
/// of that young list.
unsafe fn alloc_co_consts_array(len: usize) -> *mut FixedObjectArray {
    let payload = pyre_object::FIXED_ARRAY_ITEMS_OFFSET + len * std::mem::size_of::<PyObjectRef>();
    let raw = pyre_object::gc_hook::try_gc_alloc_nursery_raw(
        pyre_object::PY_OBJECT_ARRAY_GC_TYPE_ID,
        payload,
    );
    if raw.is_null() {
        let zeros = vec![pyre_object::PY_NULL; len];
        return unsafe { pyre_object::alloc_mro_block_gc(&zeros) };
    }
    let block = raw as *mut FixedObjectArray;
    unsafe {
        (*block).len = len;
        let items = (*block).items_mut_ptr();
        for i in 0..len {
            items.add(i).write(pyre_object::PY_NULL);
        }
    }
    block
}

/// Wrap constant `idx` of the code wrapper pinned at `code_slot`
/// (`assemble.py add_const` wraps every constant before
/// `PyCode.__init__`). A nested code constant inherits the enclosing
/// wrapper's unit. Collects.
///
/// # Safety
/// The slot must hold a `PyCode` whose `code_ptr` is readable and whose
/// constant table has more than `idx` entries.
unsafe fn realize_code_constant(code_slot: usize, idx: usize) -> PyObjectRef {
    // `box_code_constant_inheriting_unit` snapshots what it reads from the
    // parent before it allocates; nothing reads `w_code` afterwards.
    let w_code = unsafe { &*(pyre_object::gc_roots::shadow_stack_get(code_slot) as *const PyCode) };
    let code = unsafe { &*(w_code.code_ptr as *const crate::CodeObject) };
    match &crate::pyframe::code_constants(code)[idx] {
        crate::bytecode::ConstantData::Code { code } => unsafe {
            box_code_constant_inheriting_unit(&**code as *const crate::CodeObject, w_code)
        },
        constant => crate::pyframe::pyobject_from_constant(constant),
    }
}

#[majit_macros::dont_look_inside]
fn w_code_new_owned(
    code_ptr: *const (),
    hidden_applevel: bool,
    owner: usize,
    interned_name_slots: &[usize],
    realize_consts: bool,
) -> PyObjectRef {
    // RPython pointer alignment idiom (`rpython/memory/gc/minimarkpage.py:159
    // ll_assert((nsize & (WORD-1)) == 0, "malloc: size is not aligned")`):
    // bitwise AND of `cast_ptr_to_int(p)` against `(power_of_two_align - 1)`
    // gives the misalignment residual.  `front::mir` lowers a `(Ref, Int)`
    // cast to the `cast_ptr_to_int` op, so casting through `i64` (not
    // `usize`) routes the pointer through the proper LL conversion.
    // `align_of::<T>()` is always a power of two — `& (align - 1)` is
    // equivalent to `% align` for power-of-two alignments and matches the
    // RPython pattern bit-for-bit.  The residual is computed once and shared
    // by every field initializer below.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    let code_ptr_aligned = !code_ptr.is_null() && (code_ptr as i64) & align_mask == 0;
    if code_ptr_aligned {
        // The compiler emits GET_ITER yield-from mode 1; resolve it to the
        // coro-check / no-check mode the eval loop executes (`fixup_get_iter`).
        unsafe { fixup_code_get_iter(&mut *(code_ptr as *mut crate::CodeObject)) };
        // The expanded `locations` array is redundant with `linetable`; hand it
        // to [`code_locations`] to rebuild on the first reader that wants it.
        // `first_line_number` is the value the array was decoded against here;
        // a `co_firstlineno_raw` stamp that cannot be spelled as `OneIndexed`
        // corrects the record in `box_code_object_with_firstlineno`.
        let firstlineno_raw = unsafe { &*(code_ptr as *const crate::CodeObject) }
            .first_line_number
            .map(|line| line.get() as i32)
            .unwrap_or(1);
        release_code_locations(code_ptr as *mut crate::CodeObject, firstlineno_raw);
    }
    let mut fast_natural_arity = if !code_ptr_aligned {
        crate::gateway::HOPELESS
    } else {
        compute_flatcall(unsafe { &*(code_ptr as *const crate::CodeObject) })
    };
    if code_ptr_aligned
        && code_yields_inside_try(unsafe { &*(code_ptr as *const crate::CodeObject) })
    {
        fast_natural_arity |= YIELDS_INSIDE_TRY_BIT;
    }
    // `pycode.py self._globals_caches = [None] * len(self.co_names_w)`.
    let globals_caches = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<
            Option<std::sync::Weak<parking_lot::Mutex<pyre_object::celldict::GlobalCache>>>,
        > = Vec::with_capacity(names_len);
        v.resize_with(names_len, || None);
        Box::into_raw(Box::new(parking_lot::Mutex::new(v)))
    };
    // `mapdict.py self._mapdict_caches = [INVALID_CACHE_ENTRY] *
    // len(co_names_w)` — `None` is `INVALID_CACHE_ENTRY`.
    let mapdict_caches = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<Option<crate::objspace::std::mapdict::MapdictCacheEntry>> =
            Vec::with_capacity(names_len);
        v.resize_with(names_len, || None);
        Box::into_raw(Box::new(v))
    };
    // `pycode.py self.co_consts_w = consts`. The table is a young
    // `FixedObjectArray`; allocate it only after the old-gen wrapper is
    // rooted, then store with a write barrier. Holding the nursery
    // pointer in a raw local across `malloc_typed_stable` left it
    // unforwarded.
    let consts_len = if !code_ptr_aligned {
        0
    } else {
        unsafe { &*(code_ptr as *const crate::CodeObject) }
            .constants
            .len()
    };
    // `pycode.py` `PyCode.__init__`: `co_names_w = [space.new_interned_str(aname)
    // for aname in names]`. The table is main's `FixedObjectArray`; the names
    // are interned here, not on the first `getname_w`.
    let names_len = if !code_ptr_aligned {
        0
    } else {
        unsafe { &*(code_ptr as *const crate::CodeObject) }
            .names
            .len()
    };
    let npure_cellvars = if !code_ptr_aligned {
        u32::MAX
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        crate::pyframe::npure_cellvars(code_ref) as u32
    };
    let (co_nlocals, co_stacksize, co_nfreevars, co_flags) = if !code_ptr_aligned {
        (0, 0, 0, 0)
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        (
            code_ref.varnames.len() as u32,
            code_ref.max_stackdepth,
            code_ref.freevars.len() as u32,
            code_ref.flags.bits(),
        )
    };
    // `pycode.py` `PyCode._initialize`, `self.cell_families = [CellFamily(name) for name in
    // cellvars]`, laid out by localsplus slot — see the field's doc.
    // pycode.py `PyCode._initialize` leaves the list empty when the code has no cellvars.
    let cell_families = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        if code_ref.cellvars.is_empty() {
            std::ptr::null_mut()
        } else {
            let nvarnames = code_ref.varnames.len();
            let mut slots: Vec<*const CellFamily> =
                vec![std::ptr::null(); nvarnames + npure_cellvars as usize];
            let mut pure = 0usize;
            for cellvar in code_ref.cellvars.iter() {
                let cellname: &str = cellvar.as_ref();
                let family = Box::into_raw(Box::new(CellFamily::new(cellname.to_string())))
                    as *const CellFamily;
                let shared = code_ref.varnames.iter().position(|varname| {
                    let varname: &str = varname.as_ref();
                    varname == cellname
                });
                match shared {
                    Some(slot) => slots[slot] = family,
                    None => {
                        slots[nvarnames + pure] = family;
                        pure += 1;
                    }
                }
            }
            Box::into_raw(Box::new(slots))
        }
    };
    let co_firstlineno_raw = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        1
    } else {
        unsafe { &*(code_ptr as *const crate::CodeObject) }
            .first_line_number
            .map_or(1, |line| line.get() as i32)
    };
    // `pycode.py PyCode.__init__`: `self.w_filename = space.newfilename(filename)`
    // before the code object is published. Realize from `source_path` (the
    // construction-time spelling) and root it across `malloc_typed_stable`,
    // which is a `gc_op` wait point: a struct-local GCREF would not be
    // forwarded. `set_filename_bytes` re-realizes if a later writer supplies
    // the filesystem bytes `importing.py update_code_filenames` would store.
    let filename_roots = pyre_object::gc_roots::push_roots();
    let filename_slot = filename_roots.base();
    if code_ptr_aligned {
        let w_filename =
            unsafe { realize_w_filename(std::ptr::null(), code_ptr as *const crate::CodeObject) };
        let _ = filename_roots.pin_root(w_filename);
    }
    let obj = PyCode {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(&CODE_TYPE),
        },
        code_ptr: code_ptr as *const crate::CodeObject,
        co_firstlineno_raw,
        filename_bytes: std::ptr::null_mut(),
        co_code_bytes: std::ptr::null_mut(),
        filename_inherits_to_nested: false,
        w_globals: pyre_object::PY_NULL,
        w_weakreflifeline: pyre_object::PY_NULL,
        hidden_applevel,
        fast_natural_arity,
        npure_cellvars,
        co_nlocals,
        co_stacksize,
        co_nfreevars,
        co_flags,
        cell_families,
        globals_caches,
        mapdict_caches,
        co_consts_w: std::ptr::null_mut(),
        co_names_w: std::ptr::null_mut(),
        w_qualname: pyre_object::PY_NULL,
        w_name: pyre_object::PY_NULL,
        w_filename: pyre_object::PY_NULL,
        addr2line_memo: std::array::from_fn(|_| {
            std::sync::atomic::AtomicI64::new(ADDR2LINE_MEMO_EMPTY)
        }),
        loop_header_info: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
        w_globals_watchers: pyre_object::quasiimmut::QuasiImmutField::new(),
        jit_shape: std::sync::atomic::AtomicU8::new(JIT_SHAPE_UNCOMPUTED),
    };
    // The raw-address JIT compatibility seam described above requires a
    // stable wrapper address. `malloc_typed_stable` is still a managed
    // allocation: unlike the prebuilt allocator it can be reclaimed by a
    // major collection, but it is born in oldgen and never relocates.
    let obj = pyre_object::lltype::malloc_typed_stable(obj) as PyObjectRef;
    // PyPy's ast compiler has already wrapped every entry before PyCode.__init__
    // (`assemble.py add_const`). Pin the freshly allocated wrapper while
    // recursive code constants and managed scalar constants allocate, then
    // publish one object in every co_consts_w slot before returning PyCode.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let obj = pyre_object::gc_roots::pin_root(obj);
    if code_ptr_aligned {
        let w_filename = pyre_object::gc_roots::shadow_stack_get(filename_slot);
        let published = publish_code_slot_store_rooting(
            pyre_object::gc_roots::shadow_stack_get(obj_slot),
            &[w_filename],
        );
        unsafe {
            (*(published.owner() as *mut PyCode)).w_filename = published.get(0);
        }
    }
    // Claim the graph before the fill loop: each nested constant publishes its
    // own wrapper against the same owner, and a wrapper that dies during the
    // loop must not find the count at zero.
    attach_code_unit_wrapper(owner, code_ptr);
    // Both tables are young `FixedObjectArray`s (`pycode.py PyCode.__init__`
    // `co_consts_w` / `co_names_w`). A null/unaligned `code_ptr` or an empty
    // `names` list still gets a zero-length array so the field is never null.
    // One barrier remembers the old wrapper; a second barrier is a no-op once
    // TRACK_YOUNG_PTRS is clear, so both field stores follow the same barrier.
    let names_table = unsafe { alloc_co_consts_array(names_len) };
    let names_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(names_table as pyre_object::PyObjectRef);
    // `pycode.py self.co_names_w = [space.new_interned_str(aname) for aname in names]`.
    // Interning allocates, so each store reloads the table from its root slot.
    for index in 0..names_len {
        // `pycode.py` `PyCode.__init__` stores `new_interned_str`.
        // Marshal already produced the `str` (`str_from_value`); one
        // intern-table lookup keeps a live interned identity or
        // publishes the immortal miss. Interning the copied `String`
        // after a separate interned-ness probe looked the same value
        // up a second time.
        let prebuilt = (interned_name_slots.len() == names_len)
            .then(|| pyre_object::gc_roots::shadow_stack_get(interned_name_slots[index]));
        let realized = match prebuilt {
            Some(obj) if !obj.is_null() => unsafe {
                pyre_object::unicodeobject::intern_existing_str(obj)
            },
            _ => {
                let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
                pyre_object::unicodeobject::intern_str_value(code_ref.names[index].as_ref())
            }
        };
        let names_table =
            pyre_object::gc_roots::shadow_stack_get(names_slot) as *mut FixedObjectArray;
        unsafe { (*names_table).set_ref(index, realized) };
    }
    let table = unsafe { alloc_co_consts_array(consts_len) };
    let table_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(table as pyre_object::PyObjectRef);
    // Marshal fills `co_consts_w` from the decoded objects
    // (`w_code_fill_wrapped_consts`); realizing compiler-boundary
    // placeholders here would intern each string constant again and
    // then throw the wrappers away.
    if realize_consts {
        for index in 0..consts_len {
            let realized = unsafe { realize_code_constant(obj_slot, index) };
            let table =
                pyre_object::gc_roots::shadow_stack_get(table_slot) as *mut FixedObjectArray;
            unsafe { (*table).set_ref(index, realized) };
        }
    }
    publish_code_slot_store(pyre_object::gc_roots::shadow_stack_get(obj_slot));
    let owner = pyre_object::gc_roots::shadow_stack_get(obj_slot) as *mut PyCode;
    unsafe {
        (*owner).co_consts_w =
            pyre_object::gc_roots::shadow_stack_get(table_slot) as *mut FixedObjectArray;
        (*owner).co_names_w =
            pyre_object::gc_roots::shadow_stack_get(names_slot) as *mut FixedObjectArray;
    }
    pyre_object::gc_roots::shadow_stack_get(obj_slot)
}

/// pypy/interpreter/pycode.py `PyCode.__init__` shorthand —
/// equivalent to PyPy `hidden_applevel=False` default
/// (pycode.py).  Most user-level pycode constructions take this
/// path; only the gateway / continuation / `__pypy__.hidden_applevel`
/// surfaces flip the flag to `True`.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a `CodeObject` that is never freed
/// and never moved. `pycode_destructor` releases the wrapper's side tables
/// but never `code_ptr`, so a leaked box and a constant reached through one
/// both qualify.
pub fn w_code_new(code_ptr: *const ()) -> PyObjectRef {
    w_code_new_with_hidden_applevel(code_ptr, false)
}

/// `generator.py:24-27` — read the code object's cached
/// `CO_YIELD_INSIDE_TRY` equivalent.
///
/// # Safety
/// `w_code` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_yields_inside_try(w_code: PyObjectRef) -> bool {
    unsafe { (*(w_code as *const PyCode)).fast_natural_arity & YIELDS_INSIDE_TRY_BIT != 0 }
}

/// Box a compiler code object the caller owns into a heap Python code wrapper.
///
/// PyPy's compiler constructs `PyCode` directly (`pycode.py`) and
/// therefore has no foreign compiler-object in the translated graph. Pyre's
/// compiler-core `CodeObject` is a dependency ADT that recursively owns boxed
/// slices and nested constants; it is solely the serialization/API seam used
/// to reach the interpreter-level `PyCode`. Residualize that foreign seam so
/// the translated graph retains PyPy's direct `PyCode` value shape instead of
/// inventing an RPython layout for the opaque Rust value. Keep Box ownership
/// transfer and wrapper publication in the one boundary.
///
/// The object is leaked on purpose: code wrappers are immortal and
/// `pycode_destructor` never frees `code_ptr`.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the body
/// `Box::into_raw`s a `CodeObject` (unlifted raw allocation) before forwarding
/// to the residualised `w_code_new`.
#[majit_macros::dont_look_inside]
pub fn box_code_object(code: crate::CodeObject) -> PyObjectRef {
    box_code_object_with_hidden_applevel(code, false)
}

/// [`box_code_object`] when marshal already interned `co_names`.
///
/// `name_slots` are shadow-stack indices of those interned strings, in
/// `co_names` order. `pycode.py` `PyCode.__init__` would call
/// `new_interned_str` on the same characters; a slot that is already that
/// interned object is stored as-is. Compiler-boundary constants stay empty
/// so the caller can store the decoded objects (`w_code_fill_wrapped_consts`).
pub fn box_code_object_with_interned_name_slots(
    code: crate::CodeObject,
    name_slots: &[usize],
) -> PyObjectRef {
    let code_ptr = Box::into_raw(Box::new(code)) as *const ();
    w_code_new_owned(code_ptr, false, code_ptr as usize, name_slots, false)
}

/// [`box_code_object`] for the unit `gateway.py`'s `ApplevelClass` compiles,
/// which carries `hidden_applevel=True` through its whole constants graph.
#[majit_macros::dont_look_inside]
pub fn box_code_object_with_hidden_applevel(
    code: crate::CodeObject,
    hidden_applevel: bool,
) -> PyObjectRef {
    let code_ptr = Box::into_raw(Box::new(code)) as *const ();
    w_code_new_owned(code_ptr, hidden_applevel, code_ptr as usize, &[], true)
}

/// [`box_code_object`] for a caller that only has a borrow, which has to copy.
/// A caller that owns its `CodeObject` should hand it over instead — the copy
/// is a whole recursive duplicate of the constants graph.
///
/// The marshal decode path used to be such a caller twice over
/// (`make_runtime_code`, `make_code`), copying and then dropping the original;
/// both now pass ownership to [`box_code_object`]. What is left are two
/// defensive sites — [`crate::eval`]'s `code_constant` and the blackhole's
/// constant realization — that see a code object only as an element of a
/// *container* constant. The compiler does not emit that shape: the pinned
/// compiler crate's `constant_data_to_ast_constant_value` treats
/// `ConstantData::Code` as `unreachable!`, since a code object is not an
/// `ast.Constant` value. A top-level code constant reaches
/// [`box_code_constant_in_place`] instead, so the copy here is not on any
/// measured path.
///
/// Two ways to remove the copy were considered and both cost more than it:
/// widening [`box_code_constant_in_place`] would put its "never freed, never
/// moved" contract on a `pub fn` that other crates and tests call with
/// stack-owned values, and holding constants as `Rc<CodeObject>` would fork
/// `CodeObject` out of the shared compiler crate — the crate that keeps the
/// translation and interpreter halves separate.
#[majit_macros::dont_look_inside]
pub fn box_code_constant(code: &crate::CodeObject) -> PyObjectRef {
    box_code_object(code.clone())
}

/// Publish a code wrapper over a `CodeObject` the caller keeps alive, rather
/// than over a copy of it.
///
/// [`box_code_constant`] copies because most callers hold a `CodeObject` on
/// the stack. A nested compiler constant is not one of those: it sits in its
/// enclosing `CodeObject`'s constants table behind its own `Box`, that table
/// is only ever edited in place, and the enclosing object is itself leaked.
/// So the constant already outlives every wrapper published for it.
///
/// # Safety
/// `code` must point to a `CodeObject` that is never freed and never moved.
unsafe fn box_code_constant_in_place(
    code: *const crate::CodeObject,
    hidden_applevel: bool,
    owner: usize,
) -> PyObjectRef {
    w_code_new_owned(code as *const (), hidden_applevel, owner, &[], true)
}

/// Wrap a nested compiler constant and inherit the two things the enclosing
/// `PyCode` holds for a whole compilation unit rather than for one object: the
/// raw filename, when it belongs to the set selected by
/// `importing.py update_code_filenames`' `oldname` guard, and
/// `hidden_applevel`, which `assemble.py make_code` reads off `compile_info`
/// for every code object one compilation produces.
///
/// # Safety
/// `code` must satisfy [`box_code_constant_in_place`], and `parent` must be
/// the `PyCode` whose constants table holds it.
unsafe fn box_code_constant_inheriting_unit(
    code: *const crate::CodeObject,
    parent: &PyCode,
) -> PyObjectRef {
    let owner = code_unit_owner(parent.code_ptr as *const ());
    let hidden_applevel = parent.hidden_applevel;
    let inherited_filename = if parent.filename_inherits_to_nested
        && unsafe { &*code }.source_path
            == unsafe { &*(parent.code_ptr as *const crate::CodeObject) }.source_path
        && !parent.filename_bytes.is_null()
    {
        Some(unsafe { (&*parent.filename_bytes).clone() })
    } else {
        None
    };
    // Boxing can collect and move `parent`. Snapshot every value needed from
    // it first, just as RPython's GC transform spills live fields before the
    // allocation rather than retaining a native reference into the object.
    let mut obj = unsafe { box_code_constant_in_place(code, hidden_applevel, owner) };
    if let Some(bytes) = inherited_filename {
        pyre_object::with_roots!(obj => unsafe { set_filename_bytes(obj, Some(bytes)) });
        unsafe { (*(obj as *mut PyCode)).filename_inherits_to_nested = true };
        obj
    } else {
        obj
    }
}

/// Attach the filesystem bytes a whole compilation unit was named with.
///
/// `compiling.py filename='fsencode'` names the unit, not one object, so
/// recurse through the already-wrapped constants exactly like
/// `importing.py update_code_filenames`. That is the difference from
/// the code constructor and `replace`, whose filename changes only the object
/// being built and leaves nested constants on the name they compiled under.
pub unsafe fn set_compilation_unit_filename_bytes(w_code: PyObjectRef, bytes: Option<Vec<u8>>) {
    unsafe { set_compilation_unit_filename_bytes_oldname(w_code, bytes, None) }
}

/// `importing.py update_code_filenames(space, code_w, pathname, oldname=None)`.
/// This code's own filename is rewritten first, so nested constants are
/// selected by the captured original spelling.
unsafe fn set_compilation_unit_filename_bytes_oldname(
    mut w_code: PyObjectRef,
    bytes: Option<Vec<u8>>,
    oldname: Option<&[u8]>,
) {
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.pin_roots(&[w_code]);
    w_code = roots.get(code_slot);
    {
        let pycode = unsafe { &*(w_code as *const PyCode) };
        if let Some(bytes) = bytes.as_ref()
            && !pycode.code_ptr.is_null()
        {
            let same = if pycode.filename_bytes.is_null() {
                unsafe { &*(pycode.code_ptr as *const crate::CodeObject) }
                    .source_path
                    .as_bytes()
                    == bytes.as_slice()
            } else {
                unsafe { &*pycode.filename_bytes }.as_slice() == bytes.as_slice()
            };
            // Nested codes compiled with this path already spell it. Rewriting
            // each one allocated a fresh filename copy per code object.
            if same {
                return;
            }
        }
    }
    let old_filename = {
        let current = unsafe { code_filename_bytes(roots.get(code_slot)) };
        w_code = roots.get(code_slot);
        if let Some(old) = oldname
            && current.as_slice() != old
        {
            return;
        }
        current
    };
    let child_bytes = bytes.clone();
    w_code = roots.get(code_slot);
    let inherits = bytes.is_some();
    unsafe { set_filename_bytes(roots.get(code_slot), bytes) };
    w_code = roots.get(code_slot);
    unsafe {
        (*(roots.get(code_slot) as *mut PyCode)).filename_inherits_to_nested = inherits;
    }
    let mut i = 0usize;
    loop {
        let mut nested = {
            let pycode = unsafe { &*(roots.get(code_slot) as *const PyCode) };
            if pycode.co_consts_w.is_null() {
                break;
            }
            let table = unsafe { &*pycode.co_consts_w };
            if i >= table.len() {
                break;
            }
            table[i]
        };
        i += 1;
        if nested.is_null() || !unsafe { is_code(nested) } {
            continue;
        }
        let nested_scope = pyre_object::gc_roots::push_roots();
        let nested_slot = nested_scope.pin_roots(&[nested]);
        nested = nested_scope.get(nested_slot);
        if unsafe { code_filename_bytes(nested_scope.get(nested_slot)) } == old_filename {
            let child = child_bytes.clone();
            unsafe {
                set_compilation_unit_filename_bytes_oldname(
                    nested_scope.get(nested_slot),
                    child,
                    Some(old_filename.as_slice()),
                )
            };
        }
        nested = nested_scope.get(nested_slot);
        let _ = nested;
        w_code = roots.get(code_slot);
    }
}

/// Replace the owned raw filename allocation and re-realize `w_filename`.
/// `importing.py update_code_filenames` stores `code_w.co_filename = pathname`
/// then `code_w.w_filename = space.newfilename(pathname)`; this is that pair.
/// Realization allocates, so `obj` is rooted across it and reloaded.
unsafe fn set_filename_bytes(mut obj: PyObjectRef, bytes: Option<Vec<u8>>) {
    let roots = pyre_object::gc_roots::push_roots();
    let obj_slot = roots.pin_roots(&[obj]);
    obj = roots.get(obj_slot);
    {
        let slot = unsafe { &mut (*(obj as *mut PyCode)).filename_bytes };
        if let Some(bytes) = bytes {
            if slot.is_null() {
                *slot = Box::into_raw(Box::new(bytes));
            } else {
                unsafe { **slot = bytes };
            }
        } else if !slot.is_null() {
            unsafe { drop(Box::from_raw(*slot)) };
            *slot = std::ptr::null_mut();
        }
    }
    obj = roots.get(obj_slot);
    let (filename_bytes, code_ptr) = unsafe {
        let pycode = &*(obj as *const PyCode);
        (pycode.filename_bytes, pycode.code_ptr)
    };
    let w_filename = unsafe { realize_w_filename(filename_bytes, code_ptr) };
    let published = publish_code_slot_store_rooting(roots.get(obj_slot), &[w_filename]);
    unsafe {
        (*(published.owner() as *mut PyCode)).w_filename = published.get(0);
    }
    obj = roots.get(obj_slot);
    let _ = obj;
}

/// Replace the byte-exact `PyCode.co_code` fallback owned by `obj`.
///
/// A null slot means every opcode is representable by compiler-core and its
/// canonical `original_bytes()` is authoritative.
pub(crate) unsafe fn set_co_code_bytes(obj: PyObjectRef, bytes: Option<Vec<u8>>) {
    let slot = unsafe { &mut (*(obj as *mut PyCode)).co_code_bytes };
    if let Some(bytes) = bytes {
        if slot.is_null() {
            *slot = Box::into_raw(Box::new(bytes));
        } else {
            unsafe { **slot = bytes };
        }
    } else if !slot.is_null() {
        unsafe { drop(Box::from_raw(*slot)) };
        *slot = std::ptr::null_mut();
    }
}

/// Return the public, byte-exact `co_code` spelling for a live `PyCode`.
///
/// # Safety
/// `w_code` must point to a live [`PyCode`].
pub(crate) unsafe fn code_bytes(w_code: PyObjectRef) -> Vec<u8> {
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if pycode.co_code_bytes.is_null() {
        let code = unsafe { &*(pycode.code_ptr as *const crate::CodeObject) };
        code.instructions.original_bytes()
    } else {
        unsafe { (&*pycode.co_code_bytes).clone() }
    }
}

pub(crate) unsafe fn code_filename_bytes(w_code: PyObjectRef) -> Vec<u8> {
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if pycode.filename_bytes.is_null() {
        let code = unsafe { &*(pycode.code_ptr as *const crate::CodeObject) };
        code.source_path.as_bytes().to_vec()
    } else {
        unsafe { (&*pycode.filename_bytes).clone() }
    }
}

fn box_code_object_with_firstlineno(code: crate::CodeObject, firstlineno: i32) -> PyObjectRef {
    let obj = box_code_object(code);
    unsafe {
        (*(obj as *mut PyCode)).co_firstlineno_raw = firstlineno;
        // The stamp is one of the two inputs `addr2line_memo` caches against.
        for slot in &(*(obj as *mut PyCode)).addr2line_memo {
            slot.store(ADDR2LINE_MEMO_EMPTY, std::sync::atomic::Ordering::Relaxed);
        }
    }
    // `w_code_new` recorded `first_line_number`, which drops the zero and
    // negative values this stamp carries; re-record the exact one so the
    // released rows decode against what the array was built from.
    let code_ptr = unsafe { w_code_get_ptr(obj) } as *mut crate::CodeObject;
    if !code_ptr.is_null() {
        record_deferred_locations_firstlineno(code_ptr, firstlineno);
    }
    obj
}

/// Fill a newly-created code object's `co_consts_w` from the wrapped tuple
/// supplied to `CodeType.__new__` / `code.replace`.
///
/// PyPy passes this list straight into `PyCode.__init__`
/// (`pycode.py self.co_consts_w = consts`), preserving every wrapper's
/// identity. Pyre additionally serializes the values into compiler
/// `ConstantData` for bytecode decoding, but that representation must not
/// replace the interpreter-level owner.
unsafe fn w_code_fill_consts_from_tuple(obj: PyObjectRef, constants: PyObjectRef) {
    let code = unsafe { &*(obj as *const PyCode) };
    if code.co_consts_w.is_null() {
        return;
    }
    let count = unsafe { (&*code.co_consts_w).len() }.min(pyre_object::w_tuple_len(constants));
    if count == 0 {
        return;
    }
    let published = publish_code_slot_store_rooting(obj, &[constants]);
    // `ll_getitem_fast` on the const-slot table, not `Enumerate`.
    let mut index = 0usize;
    while index < count {
        let constants = published.get(0);
        if let Some(value) = unsafe { pyre_object::w_tuple_getitem(constants, index as i64) } {
            let table = unsafe { (*(published.owner() as *const PyCode)).co_consts_w };
            unsafe { (&mut *table).set_ref(index, value) };
        }
        index += 1;
    }
}

/// Record a store into a `PyCode` field the collector traces off the
/// wrapper — `co_consts_w` (the array pointer), `w_globals`, `w_filename`,
/// the mapdict method cache.
///
/// Item stores into `co_consts_w` use `FixedObjectArray::set_ref` and
/// write-barrier the array itself. This helper write-barriers the wrapper
/// for fields that are not their own GC object. A prebuilt wrapper is
/// covered by the root walk, which clean minor collections skip, hence
/// `mark_prebuilt_roots_dirty`.
///
/// Call this BEFORE the store, the way `framework.py transform_generic_set`
/// emits the `write_barrier_ptr` `direct_call` ahead of the `setfield` it
/// guards. Storing first leaves a window in which a collection sees the new
/// child through the slot while the slot is still absent from the remembered
/// set, so the collector neither traces nor rewrites it and the wrapper keeps
/// a stale pointer. A barrier that turns out to guard no store is merely
/// conservative, which is the direction upstream errs in too.
#[inline]
fn publish_code_slot_store(obj: PyObjectRef) {
    if obj.is_null() {
        return;
    }
    // The owner is a `malloc_typed_stable` / `try_gc_alloc_stable_raw`
    // PyCode. The managed barrier skips the hybrid-heap ownership lookup
    // `try_gc_write_barrier` still pays, and is the one every other
    // stable allocator uses after writing a young child.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj as pyre_object::gc_hook::GCREF);
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
}

/// `code_replace` and `unmarshal_pycode` both store a non-UTF-8 `co_name` /
/// `co_qualname` on `PyCode.w_name` / `w_qualname` (`PyCode.__init__`'s
/// `self.co_name = name`). `None` leaves the slot for `w_code_name_obj` to
/// realize from the compiler `String`.
pub(crate) fn install_code_name_objects(
    mut code: PyObjectRef,
    name: Option<PyObjectRef>,
    qualname: Option<PyObjectRef>,
) -> PyObjectRef {
    if let Some(w_name) = name {
        let published = publish_code_slot_store_rooting(code, &[w_name]);
        code = published.owner();
        unsafe {
            (*(code as *mut PyCode)).w_name = published.get(0);
        }
    }
    if let Some(w_qualname) = qualname {
        let published = publish_code_slot_store_rooting(code, &[w_qualname]);
        code = published.owner();
        unsafe {
            (*(code as *mut PyCode)).w_qualname = published.get(0);
        }
    }
    code
}

/// The words a code-slot store is about to publish, reloaded after the
/// barrier that guards it.
struct RootedSlotStore {
    scope: pyre_object::gc_roots::RootScope,
    base: usize,
}

impl RootedSlotStore {
    /// The code object that owns the slot, reloaded after the barrier.
    #[inline]
    fn owner(&self) -> PyObjectRef {
        self.scope.get(self.base)
    }

    /// The `index`th word handed to [`publish_code_slot_store_rooting`], as it
    /// stands now.
    #[inline]
    fn get(&self, index: usize) -> PyObjectRef {
        self.scope.get(self.base + 1 + index)
    }
}

/// [`publish_code_slot_store`] with the words the store is about to publish
/// held live across it.
///
/// The barrier reaches `gc_op_with_root`, which is a collection point: it can
/// wait behind another thread's collection or drive one itself. That
/// collection rewrites what it can name — the owner, through the barrier's own
/// root, and a translated `setfield`'s `newvalue`, which is a shadow-stack
/// livevar where `framework.py transform_generic_set` emits the call. A
/// `PyObjectRef` that lives only in a Rust local is named by nothing, so a
/// caller that stores one after the barrier would publish a pre-collection
/// address. Publish those words here and read them back.
fn publish_code_slot_store_rooting(obj: PyObjectRef, children: &[PyObjectRef]) -> RootedSlotStore {
    let scope = pyre_object::gc_roots::push_roots();
    let base = scope.base();
    let _ = scope.pin_root(obj);
    let children_base = scope.publish(children);
    scope.normalize(children_base, children.len());
    publish_code_slot_store(scope.get(base));
    RootedSlotStore { scope, base }
}

/// `w_code_fill_consts_from_tuple` for the marshal reader, whose `co_consts`
/// arrive as decoded objects rather than as a tuple. PyPy's marshal reader
/// passes the complete wrapped list to `PyCode.__init__`; replace every eager
/// compiler-boundary placeholder with that authoritative decoded object.
pub(crate) unsafe fn w_code_fill_wrapped_consts(obj: PyObjectRef, constants: &[PyObjectRef]) {
    let code = unsafe { &*(obj as *const PyCode) };
    if code.co_consts_w.is_null() {
        return;
    }
    let count = unsafe { (&*code.co_consts_w).len() }.min(constants.len());
    if count == 0 {
        return;
    }
    let published = publish_code_slot_store_rooting(obj, &constants[..count]);
    for index in 0..count {
        let table = unsafe { (*(published.owner() as *const PyCode)).co_consts_w };
        unsafe { (&mut *table).set_ref(index, published.get(index)) };
    }
}

/// Preserve the existing wrapped constant array when `code.replace()` changes
/// fields other than `co_consts`. PyPy copies its already-wrapped list; pyre
/// copies those same eager slot identities into the newly built wrapper.
unsafe fn w_code_copy_const_slots(dst: PyObjectRef, src: PyObjectRef) {
    let roots = pyre_object::gc_roots::push_roots();
    let root_base = roots.publish(&[dst, src]);
    roots.normalize(root_base, 2);
    let dst = roots.get(root_base);
    let src = roots.get(root_base + 1);
    let dst_code = unsafe { &*(dst as *const PyCode) };
    let src_code = unsafe { &*(src as *const PyCode) };
    if dst_code.co_consts_w.is_null() || src_code.co_consts_w.is_null() {
        return;
    }
    let n = unsafe {
        (&*dst_code.co_consts_w)
            .len()
            .min((&*src_code.co_consts_w).len())
    };
    for index in 0..n {
        let src_table = unsafe { (*(roots.get(root_base + 1) as *const PyCode)).co_consts_w };
        let value =
            unsafe { (&*src_table).load_atomic(index, std::sync::atomic::Ordering::Acquire) };
        if !value.is_null() {
            let dst_table = unsafe { (*(roots.get(root_base) as *const PyCode)).co_consts_w };
            unsafe { (&mut *dst_table).set_ref(index, value) };
        }
    }
}

/// Preserve interned `co_names_w` identities when `code.replace()` leaves
/// `co_names` alone. PyPy rebuilds through `PyCode.__init__`, which
/// `new_interned_str`s the names it was handed; those are already the
/// source slots (`fget_co_names` returns `co_names_w`). Copy the pointers
/// into the destination's existing GC table (`w_code_copy_const_slots`).
/// A null source slot is not installed over the destination's constructor
/// intern.
unsafe fn w_code_copy_name_slots(dst: PyObjectRef, src: PyObjectRef) {
    let roots = pyre_object::gc_roots::push_roots();
    let root_base = roots.publish(&[dst, src]);
    roots.normalize(root_base, 2);
    let dst = roots.get(root_base);
    let src = roots.get(root_base + 1);
    let dst_code = unsafe { &*(dst as *const PyCode) };
    let src_code = unsafe { &*(src as *const PyCode) };
    let n = unsafe {
        (&*dst_code.co_names_w)
            .len()
            .min((&*src_code.co_names_w).len())
    };
    for index in 0..n {
        let src_table = unsafe { (*(roots.get(root_base + 1) as *const PyCode)).co_names_w };
        let value =
            unsafe { (&*src_table).load_atomic(index, std::sync::atomic::Ordering::Acquire) };
        if !value.is_null() {
            let dst_table = unsafe { (*(roots.get(root_base) as *const PyCode)).co_names_w };
            unsafe { (&mut *dst_table).set_ref(index, value) };
        }
    }
}

/// `space.new_interned_w_str(w_u)` / `space.new_interned_str(aname)` for one
/// `co_names` element (`pycode.py PyCode.__init__`).
///
/// An exact `str` that is already the canonical interned object is that
/// slot (`new_interned_w_str` hits `interned_strings`). Anything else is
/// interned from its characters (`new_interned_str`), which yields an
/// interned object, never the caller's mortal string.
unsafe fn interned_co_name(w_name: PyObjectRef) -> PyObjectRef {
    if unsafe { pyre_object::pyobject::is_exact_type(w_name, &pyre_object::pyobject::STR_TYPE) }
        && unsafe { pyre_object::unicodeobject::is_interned_exact_str(w_name) }
    {
        return w_name;
    }
    pyre_object::unicodeobject::intern_wtf8_value(unsafe {
        pyre_object::unicodeobject::w_str_get_wtf8(w_name)
    })
}

/// Install `co_names_w` from the tuple `CodeType.__new__` / `code.replace`
/// was given, after the constructor has already interned `CodeObject.names`.
///
/// The tuple is the authority for identity: an element that is already the
/// canonical interned `str` stays that object (`new_interned_w_str`).
/// `code.replace` passes the source `co_names` tuple when the field is
/// overridden; omitting the field copies the source slots instead
/// ([`w_code_copy_name_slots`]). Stores go through `FixedObjectArray::set_ref`
/// on the existing GC table, the same family `w_code_fill_consts_from_tuple`
/// uses.
unsafe fn w_code_fill_names_from_tuple(obj: PyObjectRef, names: PyObjectRef) {
    if unsafe { !pyre_object::is_tuple(names) } {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    let count = unsafe { (*code.co_names_w).len() }.min(pyre_object::w_tuple_len(names));
    if count == 0 {
        return;
    }
    let published = publish_code_slot_store_rooting(obj, &[names]);
    let mut index = 0usize;
    while index < count {
        if let Some(value) = unsafe { pyre_object::w_tuple_getitem(published.get(0), index as i64) }
        {
            let interned = unsafe { interned_co_name(value) };
            let table = unsafe { (*(published.owner() as *const PyCode)).co_names_w };
            unsafe { (&mut *table).set_ref(index, interned) };
        }
        index += 1;
    }
}

/// The keyword-only fields `code.replace` accepts, in the order
/// `pypy/interpreter/pycode.py` reconstructs the code object.
const REPLACE_KWARGS: [&str; 18] = [
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_firstlineno",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_filename",
    "co_name",
    "co_qualname",
    "co_linetable",
    "co_exceptiontable",
];

/// `pycode.py` `descr_replace` packs `__args__` into `(w_args, w_kwds)`;
/// applevel `replace(self, kwds)` walks the dict (`if attr not in kwds`).
/// Named kw-only slots cannot carry a PY_NULL default (`w_dict_store`
/// rejects null values), so the bound scope is `self, **kwds`.
pub fn code_replace_signature() -> crate::gateway::Signature {
    crate::gateway::Signature::new(vec!["self"], None, Some("kwds"), 0, 1)
}

#[inline]
unsafe fn require_code(
    obj: PyObjectRef,
    descriptor: &str,
) -> Result<&'static crate::CodeObject, crate::PyError> {
    if obj.is_null() || !unsafe { is_code(obj) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{descriptor}' requires a 'code' object"
        )));
    }
    let ptr = unsafe { w_code_get_ptr(obj) } as *const crate::CodeObject;
    if ptr.is_null() || !(ptr as usize).is_multiple_of(std::mem::align_of::<crate::CodeObject>()) {
        return Err(crate::PyError::type_error("code object has no code body"));
    }
    Ok(unsafe { &*ptr })
}

/// The `co_names` / `co_varnames` / `co_freevars` / `co_cellvars` getters.
///
/// Interned for `pycode.py space.new_interned_str(aname)`' reason: the
/// entries name values, so every code object spelling one must hand back the
/// same object rather than a fresh immortal string per read.
fn names_tuple(names: &[String]) -> PyObjectRef {
    let mut items = pyre_object::gc_roots::RootedItems::new();
    for name in names {
        items.push(pyre_object::unicodeobject::intern_str_value(name));
    }
    w_tuple_new(items.take())
}

fn constants_tuple(obj: PyObjectRef, code: &crate::CodeObject) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = pyre_object::gc_roots::pin_root(obj);
    let mut constants = pyre_object::gc_roots::RootedItems::new();
    // `range` + `ll_getitem_fast`, not `Enumerate`.  `obj` is pinned:
    // `w_code_const` / `pyobject_from_constant` may collect.
    let src = crate::pyframe::code_constants(code);
    let src_p = src.as_ptr();
    let mut index = 0usize;
    while index < src.len() {
        let value =
            unsafe { w_code_const(pyre_object::gc_roots::shadow_stack_get(obj_slot), index) };
        let value = if value.is_null() {
            crate::pyframe::pyobject_from_constant(unsafe { &*src_p.add(index) })
        } else {
            value
        };
        constants.push(value);
        index += 1;
    }
    w_tuple_new(constants.take())
}

fn legacy_lnotab(code: &crate::CodeObject, firstlineno: i64) -> Vec<u8> {
    fn encode_pair(mut address: usize, mut line: i64, out: &mut Vec<u8>) {
        while address > 255 {
            out.extend_from_slice(&[255, 0]);
            address -= 255;
        }
        while line < -128 {
            out.extend_from_slice(&[address as u8, 128]);
            line += 128;
            address = 0;
        }
        while line > 127 {
            out.extend_from_slice(&[address as u8, 127]);
            line -= 127;
            address = 0;
        }
        out.extend_from_slice(&[address as u8, line as i8 as u8]);
    }

    let mut out = Vec::new();
    let mut line = firstlineno;
    let mut start_offset = 0usize;
    let mut range = PyCodeAddressRange::new(
        &code.linetable,
        firstlineno.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
    );
    while range.advance() {
        // CPython's `decode_linetable` follows the computed line rather than
        // `ar_line`, so a NO_LOCATION range does not manufacture a -1 delta.
        let next_line = range.computed_line as i64;
        if next_line != line {
            // `int bdelta = bounds.ar_start - code_offset;` is signed and
            // wraps upstream. A hand-written `co_linetable` can carry
            // `ar_end` past `i32` and hand back a start below the previous
            // one; saturating keeps the byte-delta loop below finite.
            let offset = range.ar_start as usize;
            encode_pair(
                offset.saturating_sub(start_offset),
                next_line - line,
                &mut out,
            );
            line = next_line;
            start_offset = offset;
        }
    }
    out
}

/// `PyCode.typedef` field getters. Each type-dict descriptor delegates here so
/// the object carries one authoritative compiler `CodeObject`, matching
/// `pycode.py`'s direct `co_*` attributes rather than a parallel side table.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_get_field(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if name == "co_lnotab" {
        // CPython 3.14 `code_getlnotab`: issue the warning before decoding or
        // allocating the bytes result, and propagate warnings-as-errors.
        unsafe { require_code(obj, name)? };
        // Warning dispatch can run arbitrary Python, so it is a collection
        // point.  `obj` arrives as a plain Rust local, which no root walker
        // scans: pin it on the shadow stack and read it back afterwards, then
        // reacquire the immutable CodeObject view rather than retaining a Rust
        // borrow across the re-entrant boundary.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = pyre_object::gc_roots::pin_root(obj);
        crate::warn::warn_deprecation("co_lnotab is deprecated, use co_lines instead.")?;
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let code = unsafe { require_code(obj, name)? };
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(
            &legacy_lnotab(code, unsafe {
                (*(obj as *const PyCode)).co_firstlineno_raw as i64
            }),
        ));
    }
    let code = unsafe { require_code(obj, name)? };
    Ok(match name {
        "co_argcount" => w_int_new(code.arg_count as i64),
        "co_posonlyargcount" => w_int_new(code.posonlyarg_count as i64),
        "co_kwonlyargcount" => w_int_new(code.kwonlyarg_count as i64),
        "co_nlocals" => w_int_new(code.varnames.len() as i64),
        "co_stacksize" => w_int_new(code.max_stackdepth as i64),
        "co_flags" => w_int_new(code.flags.bits() as i64),
        "co_code" | "_co_code_adaptive" => {
            pyre_object::bytesobject::w_bytes_from_bytes(&unsafe { code_bytes(obj) })
        }
        "co_consts" => constants_tuple(obj, code),
        "co_names" => names_tuple(&code.names),
        "co_varnames" => names_tuple(&code.varnames),
        "co_freevars" => names_tuple(&code.freevars),
        "co_cellvars" => names_tuple(&code.cellvars),
        "co_filename" => unsafe { w_code_filename_obj(obj) },
        // Shared realized object, like `co_qualname` below.
        "co_name" => unsafe { w_code_name_obj(obj) },
        // The realized qualname is shared with every function built from this
        // code object (`function.py self.qualname = qualname or self.name`),
        // so the attribute yields the same object on each read.
        "co_qualname" => unsafe { w_code_qualname_obj(obj) },
        "co_firstlineno" => w_int_new((*(obj as *const PyCode)).co_firstlineno_raw as i64),
        "co_linetable" => pyre_object::bytesobject::w_bytes_from_bytes(&code.linetable),
        "co_exceptiontable" => pyre_object::bytesobject::w_bytes_from_bytes(&code.exceptiontable),
        _ => {
            return Err(crate::PyError::attribute_error(format!(
                "'code' object has no attribute '{name}'"
            )));
        }
    })
}

/// CPython 3.14 `code.__new__` positional-only constructor, with the PyPy
/// `descr_code__new__` validations and field order adjusted to 3.14 (the
/// exception table precedes freevars/cellvars).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !(17..=19).contains(&args.len()) {
        return Err(crate::PyError::type_error(format!(
            "code expected at least 16 arguments, got {}",
            args.len().saturating_sub(1),
        )));
    }
    // `read_code_consts` and `box_code_object` allocate. The `&[PyObjectRef]`
    // slice is a copy of live words, so pin it and reload each slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let args_base = pyre_object::gc_roots::pin_roots(args);
    let arg = |i: usize| pyre_object::gc_roots::shadow_stack_get(args_base + i);
    let argcount = unsafe { read_code_u32(arg(1), "argcount")? };
    let posonly = unsafe { read_code_u32(arg(2), "posonlyargcount")? };
    let kwonly = unsafe { read_code_u32(arg(3), "kwonlyargcount")? };
    let nlocals = unsafe { read_code_u32(arg(4), "nlocals")? };
    let stacksize_value = unsafe { read_code_c_int(arg(5))? };
    let flags_value = unsafe { read_code_c_int(arg(6))? };
    if stacksize_value < 0 || flags_value < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Objects/codeobject.c: bad argument to internal function",
        ));
    }
    let stacksize = stacksize_value as u32;
    let flags_bits = flags_value as u32;
    let (instructions, co_code_bytes) = unsafe { read_code_units(arg(7))? };
    let constants = unsafe { read_code_consts(arg(8))? };
    let names = unsafe { read_code_names(arg(9), "names")? };
    let varnames = unsafe { read_code_names(arg(10), "varnames")? };
    if varnames.len() != nlocals as usize {
        return Err(crate::PyError::value_error(
            "code: co_nlocals != len(co_varnames)".to_string(),
        ));
    }
    let (source_path, filename_bytes) = unsafe { read_code_filename(arg(11), "filename", None)? };
    let obj_name = unsafe { read_code_str(arg(12), "name")? };
    let qualname = unsafe { read_code_str(arg(13), "qualname")? };
    let first_line = unsafe { read_code_c_int(arg(14))? } as i64;
    let first_line_number = if first_line <= 0 {
        None
    } else {
        rustpython_compiler_core::OneIndexed::new(first_line as usize)
    };
    let linetable = unsafe { read_code_bytes(arg(15), "linetable")? };
    let exceptiontable = unsafe { read_code_bytes(arg(16), "exceptiontable")? };
    let freevars = if args.len() >= 18 {
        unsafe { read_code_names(arg(17), "freevars")? }
    } else {
        Vec::<String>::new().into_boxed_slice()
    };
    let cellvars = if args.len() >= 19 {
        unsafe { read_code_names(arg(18), "cellvars")? }
    } else {
        Vec::<String>::new().into_boxed_slice()
    };
    if argcount + kwonly > nlocals || posonly > argcount {
        return Err(crate::PyError::value_error("code: invalid argument count"));
    }

    // CPython's localsplus table stores cell aliases on the local slot and
    // appends only pure cells, followed by free variables.
    let mut localspluskinds = vec![crate::bytecode::CO_FAST_LOCAL; varnames.len()];
    for cell in cellvars.iter() {
        if let Some(index) = varnames.iter().position(|name| name == cell) {
            localspluskinds[index] |= crate::bytecode::CO_FAST_CELL;
        } else {
            localspluskinds.push(crate::bytecode::CO_FAST_CELL);
        }
    }
    localspluskinds.extend(std::iter::repeat_n(
        crate::bytecode::CO_FAST_FREE,
        freevars.len(),
    ));

    // `offset2lineno` walks `co_lnotab` when a line is asked for. The expanded
    // rows are rebuilt by `code_locations` on that first reader; building them
    // here only to drop them in `release_code_locations` repeats the walk for
    // every code object import unmarshals.
    let locations = Vec::new().into_boxed_slice();
    let code = crate::CodeObject {
        instructions,
        locations,
        flags: crate::bytecode::CodeFlags::from_bits_retain(flags_bits),
        posonlyarg_count: posonly,
        arg_count: argcount,
        kwonlyarg_count: kwonly,
        source_path,
        first_line_number,
        max_stackdepth: stacksize,
        obj_name,
        qualname,
        constants,
        names,
        varnames,
        cellvars,
        freevars,
        localspluskinds: localspluskinds.into_boxed_slice(),
        linetable,
        exceptiontable,
    };
    let result = box_code_object_with_firstlineno(
        code,
        first_line.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
    );
    let mut result = result;
    pyre_object::with_roots!(result => {
        unsafe { set_filename_bytes(result, filename_bytes) };
        unsafe { set_co_code_bytes(result, co_code_bytes) };
        unsafe { w_code_fill_consts_from_tuple(result, arg(8)) };
        unsafe { w_code_fill_names_from_tuple(result, arg(9)) };
    });
    Ok(result)
}

fn code_data_equal(a: &crate::CodeObject, b: &crate::CodeObject) -> bool {
    // `co_name` / `co_qualname` are the realized str slots (`descr_code__eq__`
    // reads `self.co_name`), not the compiler `String` mirror.
    a.arg_count == b.arg_count
        && a.posonlyarg_count == b.posonlyarg_count
        && a.kwonlyarg_count == b.kwonlyarg_count
        && a.varnames.len() == b.varnames.len()
        && a.flags == b.flags
        && a.first_line_number == b.first_line_number
        && a.instructions.original_bytes() == b.instructions.original_bytes()
        && a.names.len() == b.names.len()
        && a.constants.len() == b.constants.len()
        && a.varnames == b.varnames
        && a.freevars == b.freevars
        && a.cellvars == b.cellvars
        && a.names == b.names
        && a.linetable == b.linetable
        && a.exceptiontable == b.exceptiontable
}

/// PyPy `pycode._code_const_eq` / `_convert_const`: strong equality for code
/// constants, operating on the wrapped constants owned by `PyCode.co_consts_w`.
///
/// Comparing compiler `ConstantData` here is observably wrong: the marshal
/// reader deliberately installs its decoded wrappers as the authority while
/// retaining shape-only compiler placeholders.  PyPy never looks behind
/// `co_consts_w` for equality either.
unsafe fn code_const_equal(left: PyObjectRef, right: PyObjectRef) -> Result<bool, crate::PyError> {
    // Converting the left operand allocates -- an exact string constant
    // already builds a tuple -- so hold both in slots and read the right one
    // back afterwards.
    let roots = pyre_object::gc_roots::push_roots();
    let operands = roots.publish(&[left, right]);
    roots.normalize(operands, 2);
    let mut converted = pyre_object::gc_roots::RootedItems::new();
    converted.push(_convert_const(std::ptr::null_mut(), roots.get(operands)));
    converted.push(_convert_const(
        std::ptr::null_mut(),
        roots.get(operands + 1),
    ));
    crate::baseobjspace::eq_w(converted.get(0), converted.get(1))
}

unsafe fn code_objects_equal(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<bool, crate::PyError> {
    // `w_code_const` may realize a nested code wrapper and collect.  PyPy's
    // `descr_code__eq__` keeps both `PyCode` instances live as ordinary
    // RPython references throughout the loop; reload the equivalent rooted
    // slots at every recursive code-object level.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::pin_roots(&[this, other]);
    let this = pyre_object::gc_roots::shadow_stack_get(root_base);
    let other = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    let a = unsafe { require_code(this, "__eq__")? };
    let b = unsafe { require_code(other, "__eq__")? };
    let left_name = unsafe { w_code_name_obj(pyre_object::gc_roots::shadow_stack_get(root_base)) };
    let right_name =
        unsafe { w_code_name_obj(pyre_object::gc_roots::shadow_stack_get(root_base + 1)) };
    let this = pyre_object::gc_roots::shadow_stack_get(root_base);
    let other = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
    // [3.14-spec] code equality ignores co_qualname ↔ `descr_code__eq__`
    // compares `self.co_qualname`. A qualname-only `replace` still compares equal.
    if left_name.is_null()
        || right_name.is_null()
        || !unsafe { pyre_object::w_str_eq_w(left_name, right_name) }
        || unsafe { (*(this as *const PyCode)).co_firstlineno_raw }
            != unsafe { (*(other as *const PyCode)).co_firstlineno_raw }
        || unsafe { code_bytes(this) } != unsafe { code_bytes(other) }
        || !code_data_equal(a, b)
    {
        return Ok(false);
    }
    for index in 0..a.constants.len() {
        let _constant_roots = pyre_object::gc_roots::push_roots();
        let constant_root_base = pyre_object::gc_roots::shadow_stack_len();
        let left =
            unsafe { w_code_const(pyre_object::gc_roots::shadow_stack_get(root_base), index) };
        let _ = pyre_object::gc_roots::pin_root(left);
        let right = unsafe {
            w_code_const(
                pyre_object::gc_roots::shadow_stack_get(root_base + 1),
                index,
            )
        };
        let _ = pyre_object::gc_roots::pin_root(right);
        let left = pyre_object::gc_roots::shadow_stack_get(constant_root_base);
        let right = pyre_object::gc_roots::shadow_stack_get(constant_root_base + 1);
        if left.is_null() || right.is_null() || !unsafe { code_const_equal(left, right)? } {
            return Ok(false);
        }
    }
    Ok(true)
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_eq(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_code(other) } {
        return Ok(pyre_object::special::w_not_implemented());
    }
    Ok(w_bool_from(unsafe { code_objects_equal(this, other)? }))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_ne(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let equal = unsafe { code_eq(this, other)? };
    if unsafe { pyre_object::is_not_implemented(equal) } {
        Ok(equal)
    } else {
        Ok(w_bool_from(!unsafe { w_bool_get_value(equal) }))
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_hash(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let code = unsafe { require_code(obj, "__hash__")? };
    #[inline]
    fn scramble(result: i64, value: i64) -> i64 {
        ((result as u64 ^ value as u64).wrapping_mul(1_000_003)) as i64
    }
    #[inline]
    fn add_obj(result: &mut i64, value: PyObjectRef) -> Result<(), crate::PyError> {
        *result = scramble(*result, crate::baseobjspace::hash_w_strict(value)?);
        Ok(())
    }
    let mut result = 20_250_211i64;
    // `descr_code__hash__` also hashes `co_qualname`. 3.14's code hash does
    // not, so equal qualname-only replacements keep one hash (`code_hash`).
    add_obj(&mut result, unsafe {
        w_code_name_obj(pyre_object::gc_roots::shadow_stack_get(obj_slot))
    })?;
    let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    for value in [
        code.arg_count as i64,
        code.posonlyarg_count as i64,
        code.kwonlyarg_count as i64,
        code.varnames.len() as i64,
        code.flags.bits() as i64,
        (*(obj as *const PyCode)).co_firstlineno_raw as i64,
    ] {
        result = scramble(result, value);
    }
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&unsafe { code_bytes(obj) }),
    )?;
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.linetable),
    )?;
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.exceptiontable),
    )?;
    for names in [&code.varnames, &code.freevars, &code.cellvars, &code.names] {
        for name in names.iter() {
            add_obj(&mut result, w_str_new_managed(name))?;
        }
    }
    let src = crate::pyframe::code_constants(code);
    let src_p = src.as_ptr();
    let mut index = 0usize;
    while index < src.len() {
        let w_constant = unsafe { w_code_const(obj, index) };
        let w_constant = if w_constant.is_null() {
            crate::pyframe::pyobject_from_constant(unsafe { &*src_p.add(index) })
        } else {
            w_constant
        };
        add_obj(&mut result, w_constant)?;
        index += 1;
    }
    Ok(if result == -1 { -2 } else { result })
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_repr(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let _code = unsafe { require_code(obj, "__repr__")? };
    let w_name = unsafe { w_code_name_obj(pyre_object::gc_roots::shadow_stack_get(obj_slot)) };
    let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
    // pycode.py `get_repr` / `descr_code__repr__` print `self.co_name`.
    // The zero sentinel is line -1.
    let raw_line = (*(obj as *const PyCode)).co_firstlineno_raw as i64;
    let line = if raw_line == 0 { -1 } else { raw_line };
    let mut repr = rustpython_wtf8::Wtf8Buf::new();
    repr.push_str("<code object ");
    if !w_name.is_null() {
        repr.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_name) });
    }
    repr.push_str(" at ");
    repr.push_str(&crate::display::repr_gc_addr(obj));
    repr.push_str(", file \"");
    let filename = crate::gateway::fsdecode_filename_wtf8(&unsafe { code_filename_bytes(obj) });
    repr.push_wtf8(&filename);
    repr.push_str("\", line ");
    repr.push_str(&line.to_string());
    repr.push_str(">");
    Ok(pyre_object::w_str_from_wtf8_managed(repr))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_varname_from_oparg(
    obj: PyObjectRef,
    index: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "_varname_from_oparg")? };
    let mut index = unsafe { crate::builtins::space_index_w(index)? };
    if index >= 0 {
        // closure-free, Option-pattern-free `varnames.get` / `freevars.get`
        // rewrites — keep the bounds check a plain `lt + getitem`.
        let vi = index as usize;
        if vi < code.varnames.len() {
            return Ok(w_str_new_managed(&code.varnames[vi]));
        }
        index -= code.varnames.len() as i64;
        let pure_cellvars = code
            .cellvars
            .iter()
            .filter(|cell| !code.varnames.iter().any(|var| var == *cell));
        let pure_cellvar_count = pure_cellvars.clone().count();
        if let Some(name) = pure_cellvars.skip(index as usize).next() {
            return Ok(w_str_new_managed(name));
        }
        index -= pure_cellvar_count as i64;
        let fi = index as usize;
        if fi < code.freevars.len() {
            return Ok(w_str_new_managed(&code.freevars[fi]));
        }
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::IndexError,
        "tuple index out of range",
    ))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
/// `lineiterator` — the object `co_lines()` returns.
///
/// The walk is lazy, as upstream's is: `next()` decodes one range (plus the
/// lookahead the merge needs) and no more.  Materializing the whole table up
/// front made a single `co_lines()` call on a large code object cost time
/// proportional to the table, and `traceback._get_code_position` calls
/// `co_positions()` for every frame it renders.
#[crate::pyre_class("line_iterator", static_name = "LINE_ITER")]
pub struct W_LineIterObject {
    /// `li_code` — the code object whose `co_linetable` the walk reads, held
    /// so the table outlives the iterator.
    pub w_code: PyObjectRef,
    /// `li_line`, suspended.  See [`PyCodeAddressRange::cursor`].
    pub pos: i64,
    pub ar_start: i64,
    pub ar_end: i64,
    pub ar_line: i64,
    pub computed_line: i64,
}

/// `positionsiterator` — the object `co_positions()` returns.
#[crate::pyre_class("positions_iterator", static_name = "POSITIONS_ITER")]
pub struct W_PositionsIterObject {
    /// `pi_code`.
    pub w_code: PyObjectRef,
    /// `pi_range`, suspended.
    pub pos: i64,
    pub ar_start: i64,
    pub ar_end: i64,
    pub ar_line: i64,
    pub computed_line: i64,
    /// `pi_offset`, and the three locations `advance_with_locations` writes
    /// through its out-params.
    pub offset: i64,
    pub end_line: i64,
    pub column: i64,
    pub end_column: i64,
}

/// `branchesiterator` — the object `co_branches()` returns.
///
/// `_PyBranchesIterator` reports `line_iterator` as its `tp_name`, the same
/// string `_PyLineIterator` carries, so the two distinct types are
/// indistinguishable by name; only `type(a) is type(b)` tells them apart.
#[crate::pyre_class("line_iterator", static_name = "BRANCHES_ITER")]
pub struct W_BranchesIterObject {
    /// `bi_code` — the code object whose bytecode the walk reads, held so it
    /// outlives the iterator.
    pub w_code: PyObjectRef,
    /// `bi_offset`, in code units.
    pub offset: i64,
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_line_iter(obj: PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && std::ptr::eq(unsafe { (*obj).ob_type }, &LINE_ITER_TYPE)
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_positions_iter(obj: PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && std::ptr::eq(unsafe { (*obj).ob_type }, &POSITIONS_ITER_TYPE)
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_branches_iter(obj: PyObjectRef) -> bool {
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return false;
    }
    !obj.is_null() && std::ptr::eq(unsafe { (*obj).ob_type }, &BRANCHES_ITER_TYPE)
}

/// Whether `obj` is an instance of *a* type named `line_iterator`.
///
/// The owner a builtin method reports its receiver against is keyed by type
/// name, and two types answer to `line_iterator`, so the name-keyed test
/// admits either; each method body still checks the one layout it reads.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn is_named_line_iterator(obj: PyObjectRef) -> bool {
    unsafe { is_line_iter(obj) || is_branches_iter(obj) }
}

/// The `co_linetable` an iterator walks, or `None` once its code object is
/// gone.
///
/// # Safety
/// `w_code` must be null or a live `PyCode`.
unsafe fn iter_linetable<'a>(w_code: PyObjectRef) -> Option<&'a [u8]> {
    if w_code.is_null() {
        return None;
    }
    let code = unsafe { w_code_get_ptr(w_code) } as *const crate::CodeObject;
    if code.is_null() {
        return None;
    }
    Some(unsafe { &(*code).linetable })
}

/// `lineiter_next` — one merged `(start, end, line)` range.
///
/// ```c
/// if (!_PyLineTable_NextAddressRange(bounds)) { return NULL; }
/// int start = bounds->ar_start;
/// int line = bounds->ar_line;
/// while (_PyLineTable_NextAddressRange(bounds)) {
///     if (bounds->ar_line != line) {
///         _PyLineTable_PreviousAddressRange(bounds);
///         break;
///     }
/// }
/// return Py_BuildValue("iiO&", start, bounds->ar_end,
///                      _source_offset_converter, &line);
/// ```
///
/// The lookahead is undone by restoring the saved cursor rather than by
/// `retreat`: restoring is the exact inverse of the `advance` it undoes, where
/// `retreat` re-derives the previous entry's line and extent by scanning
/// backwards and is a second decoder to keep in step.
///
/// # Safety
/// `obj` must be a live [`W_LineIterObject`].
pub unsafe fn line_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        let it = &mut *(obj as *mut W_LineIterObject);
        let linetable = iter_linetable(it.w_code)?;
        let mut bounds = PyCodeAddressRange::resume(linetable, line_iter_cursor(it));
        if !bounds.advance() {
            return None;
        }
        let start = bounds.ar_start;
        let line = bounds.ar_line;
        loop {
            let saved = bounds.cursor();
            if !bounds.advance() {
                break;
            }
            if bounds.ar_line != line {
                bounds.restore(saved);
                break;
            }
        }
        let end = bounds.ar_end;
        line_iter_store(it, bounds.cursor());
        // Each field is freshly allocated and the next one allocates again,
        // so they are pinned as they arrive (`build_list_storage`).
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(w_int_new(start as i64));
        fields.push(w_int_new(end as i64));
        fields.push(source_offset_converter(line));
        Some(w_tuple_new(fields.take()))
    }
}

fn line_iter_cursor(it: &W_LineIterObject) -> LineTableCursor {
    LineTableCursor {
        pos: it.pos as usize,
        ar_start: it.ar_start as i32,
        ar_end: it.ar_end as i32,
        ar_line: it.ar_line as i32,
        computed_line: it.computed_line as i32,
    }
}

fn line_iter_store(it: &mut W_LineIterObject, cursor: LineTableCursor) {
    it.pos = cursor.pos as i64;
    it.ar_start = cursor.ar_start as i64;
    it.ar_end = cursor.ar_end as i64;
    it.ar_line = cursor.ar_line as i64;
    it.computed_line = cursor.computed_line as i64;
}

/// `positionsiter_next` — one `(line, end_line, column, end_column)` tuple per
/// code unit, re-decoding only when the offset reaches the end of the range.
///
/// ```c
/// if (pi->pi_offset >= pi->pi_range.ar_end) {
///     if (at_end(&pi->pi_range)) { return NULL; }
///     advance_with_locations(&pi->pi_range, &pi->pi_endline, &pi->pi_column,
///                            &pi->pi_endcolumn);
/// }
/// pi->pi_offset += 2;
/// ```
///
/// # Safety
/// `obj` must be a live [`W_PositionsIterObject`].
pub unsafe fn positions_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        let it = &mut *(obj as *mut W_PositionsIterObject);
        let linetable = iter_linetable(it.w_code)?;
        let cursor = LineTableCursor {
            pos: it.pos as usize,
            ar_start: it.ar_start as i32,
            ar_end: it.ar_end as i32,
            ar_line: it.ar_line as i32,
            computed_line: it.computed_line as i32,
        };
        let mut bounds = PyCodeAddressRange::resume(linetable, cursor);
        if it.offset as i32 >= bounds.ar_end {
            let Some((line, end_line, column, end_column)) = bounds.advance_with_locations() else {
                return None;
            };
            it.ar_line = line as i64;
            it.end_line = end_line as i64;
            it.column = column as i64;
            it.end_column = end_column as i64;
        }
        let saved = bounds.cursor();
        it.pos = saved.pos as i64;
        it.ar_start = saved.ar_start as i64;
        it.ar_end = saved.ar_end as i64;
        it.ar_line = saved.ar_line as i64;
        it.computed_line = saved.computed_line as i64;
        it.offset = it.offset.wrapping_add(2);
        let mut fields = pyre_object::gc_roots::RootedItems::new();
        fields.push(source_offset_converter(it.ar_line as i32));
        fields.push(source_offset_converter(it.end_line as i32));
        fields.push(source_offset_converter(it.column as i32));
        fields.push(source_offset_converter(it.end_column as i32));
        Some(w_tuple_new(fields.take()))
    }
}

/// `code_positionsiterator` — `co_positions()`.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_positions(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe { require_code(obj, "co_positions")? };
    let _roots = pyre_object::gc_roots::push_roots();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let first_line = unsafe { (*(obj as *const PyCode)).co_firstlineno_raw };
    Ok(W_PositionsIterObject::allocate_stable(
        W_PositionsIterObject {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_code: obj,
            pos: 0,
            ar_start: -1,
            ar_end: 0,
            ar_line: -1,
            computed_line: first_line as i64,
            offset: 0,
            end_line: -1,
            column: -1,
            end_column: -1,
        },
    ))
}

/// `code_linesiterator` — `co_lines()`.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_lines(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe { require_code(obj, "co_lines")? };
    let _roots = pyre_object::gc_roots::push_roots();
    let obj = pyre_object::gc_roots::pin_root(obj);
    let first_line = unsafe { (*(obj as *const PyCode)).co_firstlineno_raw };
    Ok(W_LineIterObject::allocate_stable(W_LineIterObject {
        ob: pyre_object::PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_code: obj,
        pos: 0,
        ar_start: -1,
        ar_end: 0,
        ar_line: -1,
        computed_line: first_line as i64,
    }))
}

/// `int_triple` — the `(a, b, c)` tuple every branch row is reported as.
fn int_triple(a: i64, b: i64, c: i64) -> PyObjectRef {
    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(w_int_new(a));
    fields.push(w_int_new(b));
    fields.push(w_int_new(c));
    w_tuple_new(fields.take())
}

/// `branchesiter_next` — the next branch instruction's
/// `(instruction, not-taken target, taken target)` triple, in byte offsets.
///
/// The scan resumes at `bi_offset` and stops at the branch it reports, so a
/// caller that reads one row does not pay for the rest of the bytecode.
///
/// # Safety
/// `obj` must be a live [`W_BranchesIterObject`].
pub unsafe fn branches_iter_next(obj: PyObjectRef) -> Option<PyObjectRef> {
    unsafe {
        let it = &mut *(obj as *mut W_BranchesIterObject);
        if it.w_code.is_null() {
            return None;
        }
        let code = w_code_get_ptr(it.w_code) as *const crate::CodeObject;
        if code.is_null() {
            return None;
        }
        let code = &*code;
        let mut offset = usize::try_from(it.offset).ok()?;
        let mut op_arg = 0usize;
        while offset < code.instructions.len() {
            let op = code.instructions.read_op(offset).deoptimize();
            let next_offset = offset + 1 + op.cache_entries();
            let arg = u8::from(code.instructions.read_arg(offset)) as usize;
            match op {
                crate::bytecode::Instruction::ExtendedArg => {
                    op_arg = (op_arg << 8) | arg;
                }
                crate::bytecode::Instruction::ForIter { .. } => {
                    op_arg = (op_arg << 8) | arg;
                    it.offset = next_offset as i64;
                    // Skips END_FOR and POP_ITER.
                    let target = next_offset + op_arg + 2;
                    return Some(int_triple(
                        (offset * 2) as i64,
                        (next_offset * 2) as i64,
                        (target * 2) as i64,
                    ));
                }
                crate::bytecode::Instruction::PopJumpIfFalse { .. }
                | crate::bytecode::Instruction::PopJumpIfTrue { .. }
                | crate::bytecode::Instruction::PopJumpIfNone { .. }
                | crate::bytecode::Instruction::PopJumpIfNotNone { .. } => {
                    op_arg = (op_arg << 8) | arg;
                    // Skip NOT_TAKEN, which Python 3.14 emits at the
                    // fallthrough edge so branch instrumentation can tell the
                    // untaken path apart.
                    let not_taken = next_offset + 1;
                    it.offset = not_taken as i64;
                    return Some(int_triple(
                        (offset * 2) as i64,
                        (not_taken * 2) as i64,
                        ((next_offset + op_arg) * 2) as i64,
                    ));
                }
                crate::bytecode::Instruction::EndAsyncFor => {
                    op_arg = (op_arg << 8) | arg;
                    let src_offset = next_offset - op_arg;
                    it.offset = next_offset as i64;
                    debug_assert!(matches!(
                        code.instructions.read_op(src_offset).deoptimize(),
                        crate::bytecode::Instruction::EndSend
                    ));
                    debug_assert!(matches!(
                        code.instructions.read_op(src_offset + 1).deoptimize(),
                        crate::bytecode::Instruction::NotTaken
                    ));
                    let not_taken = src_offset + 2;
                    return Some(int_triple(
                        (src_offset * 2) as i64,
                        (not_taken * 2) as i64,
                        (next_offset * 2) as i64,
                    ));
                }
                _ => op_arg = 0,
            }
            offset = next_offset;
        }
        it.offset = offset as i64;
        None
    }
}

/// `_PyInstrumentation_BranchesIterator` — `co_branches()`.
///
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_branches(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe { require_code(obj, "co_branches")? };
    let _roots = pyre_object::gc_roots::push_roots();
    let obj = pyre_object::gc_roots::pin_root(obj);
    Ok(W_BranchesIterObject::allocate_stable(
        W_BranchesIterObject {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_code: obj,
            offset: 0,
        },
    ))
}

/// `code.replace(**kwds)` — `pypy/interpreter/pycode.py` applevel
/// `replace`, which gathers every `co_*` attribute (taking the keyword
/// override where present) and reconstructs the code object through the
/// `CodeType` constructor.  pyre stores a compiler `CodeObject`, so the
/// equivalent is to clone it, override each supplied field, and re-box it.
///
/// # Safety
/// `args[0]` must be the receiver `code` object (verified).
pub unsafe fn code_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Bound scope: `self, **kwds` (`pycode.py` descr_replace / applevel replace).
    let w_self = args
        .first()
        .copied()
        .filter(|o| !o.is_null())
        .unwrap_or(PY_NULL);
    if w_self.is_null() || !unsafe { is_code(w_self) } {
        return Err(crate::PyError::type_error(
            "descriptor 'replace' requires a 'code' object",
        ));
    }
    // `read_code_consts` and `box_code_object` allocate. Pin the bound scope
    // and reload the kwargs dict from its slot.
    let _roots = pyre_object::gc_roots::push_roots();
    let root_base = pyre_object::gc_roots::pin_roots(args);
    let n_bound = args.len();
    let w_self = || pyre_object::gc_roots::shadow_stack_get(root_base);
    if n_bound >= 2 {
        let kwds = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        if !kwds.is_null() {
            crate::builtins::kwarg_reject_unknown(Some(kwds), &REPLACE_KWARGS, "replace")?;
        }
    }
    let get = |name: &str| -> Option<PyObjectRef> {
        if n_bound < 2 {
            return None;
        }
        let kwds = pyre_object::gc_roots::shadow_stack_get(root_base + 1);
        if kwds.is_null() {
            return None;
        }
        // `pycode.py` applevel `replace`: `if attr not in kwds` inherits;
        // a supplied value, including `None`, is passed to the constructor.
        unsafe { pyre_object::w_dict_getitem_str(kwds, name) }
    };

    let code_ptr = unsafe { w_code_get_ptr(w_self()) } as *const crate::CodeObject;
    if code_ptr.is_null() {
        return Err(crate::PyError::type_error(
            "cannot replace fields of a code object with no code body",
        ));
    }
    let mut code = unsafe { (*code_ptr).clone() };
    // `code.replace` declares `co_nlocals` as
    // `int(c_default="((PyCodeObject *)self)->co_nlocals")`, so an omitted
    // keyword carries the RECEIVER's count into the constructor's
    // `co_nlocals != len(co_varnames)` check.  Read it before any override
    // runs: taking it from the already-replaced `code.varnames` would make
    // that check vacuous and accept a `co_varnames` of a different length.
    let inherited_nlocals = code.varnames.len();
    let mut firstlineno_raw = unsafe { (*(w_self() as *const PyCode)).co_firstlineno_raw };
    let mut filename_bytes = unsafe {
        let ptr = (*(w_self() as *const PyCode)).filename_bytes;
        if ptr.is_null() {
            None
        } else {
            Some((&*ptr).clone())
        }
    };
    let mut filename_inherits_to_nested =
        unsafe { (*(w_self() as *const PyCode)).filename_inherits_to_nested };
    let mut co_code_bytes = unsafe {
        let ptr = (*(w_self() as *const PyCode)).co_code_bytes;
        if ptr.is_null() {
            None
        } else {
            Some((&*ptr).clone())
        }
    };
    let rebuild_localspluskinds = get("co_varnames").is_some()
        || get("co_cellvars").is_some()
        || get("co_freevars").is_some();

    if let Some(v) = get("co_argcount") {
        code.arg_count = unsafe { read_code_u32(v, "co_argcount")? };
    }
    if let Some(v) = get("co_posonlyargcount") {
        code.posonlyarg_count = unsafe { read_code_u32(v, "co_posonlyargcount")? };
    }
    if let Some(v) = get("co_kwonlyargcount") {
        code.kwonlyarg_count = unsafe { read_code_u32(v, "co_kwonlyargcount")? };
    }
    let nlocals = match get("co_nlocals") {
        Some(v) => unsafe { read_code_u32(v, "co_nlocals")? as usize },
        None => inherited_nlocals,
    };
    if let Some(v) = get("co_stacksize") {
        code.max_stackdepth = unsafe { read_code_u32(v, "co_stacksize")? };
    }
    if let Some(v) = get("co_flags") {
        let value = unsafe { read_code_c_int(v)? };
        if value < 0 {
            return Err(crate::PyError::value_error(
                "co_flags must be a positive integer",
            ));
        }
        let bits = value as u32;
        code.flags = crate::bytecode::CodeFlags::from_bits_retain(bits);
    }
    if let Some(v) = get("co_firstlineno") {
        let n = unsafe { read_code_c_int(v)? } as i64;
        if n < 0 {
            return Err(crate::PyError::value_error(
                "co_firstlineno must be a positive integer",
            ));
        }
        firstlineno_raw = n.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        code.first_line_number = if n <= 0 {
            None
        } else {
            rustpython_compiler_core::OneIndexed::new(n as usize)
        };
    }
    // `pycode.py` `PyCode.__init__` stores one field, `self.co_name = name`
    // (and `self.co_qualname`). That field is `PyCode.w_name` /
    // `w_qualname` (`w_code_name_obj`). Every app-level reader uses it.
    // `CodeObject.obj_name` stays the compiler UTF-8 mirror JIT debug names
    // read; a lone surrogate is not written into it.
    // A lone-surrogate name is kept as a str object. `read_code_str` and the
    // field parsers below allocate, so each surrogate is published with
    // `pin_root` (`framework.py` `push_roots` writes that livevar) before the
    // next parser runs, and read back from its slot afterwards.
    let _name_roots = pyre_object::gc_roots::push_roots();
    let mut name_slot: Option<usize> = None;
    let mut qual_slot: Option<usize> = None;
    if let Some(v) = get("co_name") {
        if unsafe { pyre_object::is_str(v) } && unsafe { !pyre_object::w_str_is_utf8(v) } {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = _name_roots.pin_root(v);
            name_slot = Some(slot);
        } else {
            code.obj_name = unsafe { read_code_str(v, "co_name")? };
        }
    }
    if let Some(v) = get("co_qualname") {
        if unsafe { pyre_object::is_str(v) } && unsafe { !pyre_object::w_str_is_utf8(v) } {
            let slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = _name_roots.pin_root(v);
            qual_slot = Some(slot);
        } else {
            code.qualname = unsafe { read_code_str(v, "co_qualname")? };
        }
    }
    if let Some(v) = get("co_filename") {
        (code.source_path, filename_bytes) =
            unsafe { read_code_filename(v, "co_filename", Some(&code.source_path))? };
        filename_inherits_to_nested = false;
    }
    if let Some(v) = get("co_names") {
        code.names = unsafe { read_code_names(v, "co_names")? };
    }
    if let Some(v) = get("co_varnames") {
        code.varnames = unsafe { read_code_names(v, "co_varnames")? };
    }
    if let Some(v) = get("co_freevars") {
        code.freevars = unsafe { read_code_names(v, "co_freevars")? };
    }
    if let Some(v) = get("co_cellvars") {
        code.cellvars = unsafe { read_code_names(v, "co_cellvars")? };
    }
    if let Some(v) = get("co_linetable") {
        code.linetable = unsafe { read_code_bytes(v, "co_linetable")? };
    }
    if let Some(v) = get("co_exceptiontable") {
        code.exceptiontable = unsafe { read_code_bytes(v, "co_exceptiontable")? };
    }
    if let Some(v) = get("co_consts") {
        code.constants = unsafe { read_code_consts(v)? };
        filename_inherits_to_nested = false;
    }
    if let Some(v) = get("co_code") {
        (code.instructions, co_code_bytes) = unsafe { read_code_units(v)? };
    }

    if nlocals != code.varnames.len() {
        return Err(crate::PyError::value_error(
            "code: co_nlocals != len(co_varnames)",
        ));
    }
    if code.posonlyarg_count > code.arg_count {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Objects/codeobject.c: bad argument to internal function",
        ));
    }
    if code.arg_count as usize + code.kwonlyarg_count as usize > code.varnames.len() {
        return Err(crate::PyError::value_error(
            "code: co_varnames is too small",
        ));
    }

    // PyCode_Replace preserves the private locals-plus kind table when only
    // unrelated public fields change.  This matters for PEP 709's
    // CO_FAST_HIDDEN slots, which cannot be reconstructed from co_varnames,
    // co_cellvars and co_freevars.  Rebuild only when one of those public
    // layout fields was explicitly replaced.
    if rebuild_localspluskinds {
        let mut localspluskinds = vec![crate::bytecode::CO_FAST_LOCAL; code.varnames.len()];
        for cell in code.cellvars.iter() {
            if let Some(index) = code.varnames.iter().position(|name| name == cell) {
                localspluskinds[index] |= crate::bytecode::CO_FAST_CELL;
            } else {
                localspluskinds.push(crate::bytecode::CO_FAST_CELL);
            }
        }
        localspluskinds.extend(std::iter::repeat_n(
            crate::bytecode::CO_FAST_FREE,
            code.freevars.len(),
        ));
        code.localspluskinds = localspluskinds.into_boxed_slice();
    }
    // Same deferral as the marshal reader: `offset2lineno` does not expand
    // the line table until a reader asks. `code_locations` rebuilds it.
    code.locations = Vec::new().into_boxed_slice();

    let mut result = box_code_object_with_firstlineno(code, firstlineno_raw);
    let w_name = name_slot.map(pyre_object::gc_roots::shadow_stack_get);
    let w_qualname = qual_slot.map(pyre_object::gc_roots::shadow_stack_get);
    result = install_code_name_objects(result, w_name, w_qualname);
    let result_roots = pyre_object::gc_roots::push_roots();
    let result_slot = result_roots.pin_roots(&[result]);
    unsafe { set_filename_bytes(result_roots.get(result_slot), filename_bytes) };
    let result = result_roots.get(result_slot);
    unsafe { set_co_code_bytes(result, co_code_bytes) };
    unsafe {
        (*(result as *mut PyCode)).filename_inherits_to_nested = filename_inherits_to_nested;
    }
    if let Some(constants) = get("co_consts") {
        unsafe { w_code_fill_consts_from_tuple(result_roots.get(result_slot), constants) };
    } else {
        unsafe { w_code_copy_const_slots(result_roots.get(result_slot), w_self()) };
    }
    if let Some(names) = get("co_names") {
        unsafe { w_code_fill_names_from_tuple(result_roots.get(result_slot), names) };
    } else {
        unsafe { w_code_copy_name_slots(result_roots.get(result_slot), w_self()) };
    }
    Ok(result_roots.get(result_slot))
}

/// A non-negative `co_*` count argument as `u32`.
unsafe fn read_code_u32(v: PyObjectRef, field: &str) -> Result<u32, crate::PyError> {
    let n = unsafe { read_code_c_int(v)? } as i64;
    if n < 0 {
        let message = if field.starts_with("co_") {
            format!("{field} must be a positive integer")
        } else {
            format!("code: {field} must not be negative")
        };
        return Err(crate::PyError::value_error(message));
    }
    Ok(n as u32)
}

/// Argument Clinic converts every public code-object integer through C `int`.
unsafe fn read_code_c_int(v: PyObjectRef) -> Result<i32, crate::PyError> {
    let n = unsafe { crate::builtins::space_index_w(v)? };
    i32::try_from(n).map_err(|_| {
        crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            "Python int too large to convert to C int",
        )
    })
}

/// A `str` `co_*` field as an owned `String` (the compiler `Name` type).
unsafe fn read_code_str(v: PyObjectRef, field: &str) -> Result<String, crate::PyError> {
    if !unsafe { pyre_object::is_str(v) } {
        return Err(crate::PyError::type_error(format!("{field} must be a str")));
    }
    Ok(crate::baseobjspace::str_utf8_w(v)?.to_string())
}

/// `pycode.py filename='fsencode'`: retain filesystem bytes that the
/// compiler dependency's UTF-8 `source_path` cannot represent. A supplied
/// `fallback` is the last valid UTF-8 spelling used by interpreter/JIT readers;
/// a new code object uses a lossy compiler-only spelling until those readers
/// can consume the byte-exact field directly.
unsafe fn read_code_filename(
    v: PyObjectRef,
    field: &str,
    fallback: Option<&str>,
) -> Result<(String, Option<Vec<u8>>), crate::PyError> {
    if !unsafe { pyre_object::is_str(v) } {
        return Err(crate::PyError::type_error(format!("{field} must be a str")));
    }
    let bytes = crate::gateway::fsencode_bytes_w(v)?;
    Ok(split_code_filename_bytes(bytes, fallback))
}

/// Split the authoritative filesystem bytes from the compiler dependency's
/// UTF-8-only `source_path` spelling (`objspace.py newfilename`).
pub fn split_code_filename_bytes(
    bytes: Vec<u8>,
    fallback: Option<&str>,
) -> (String, Option<Vec<u8>>) {
    match String::from_utf8(bytes) {
        Ok(source_path) => (source_path, None),
        Err(error) => {
            let bytes = error.into_bytes();
            let source_path = fallback
                .map(str::to_owned)
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
            (source_path, Some(bytes))
        }
    }
}

/// A `tuple[str]` `co_*` field (names / varnames / freevars / cellvars).
unsafe fn read_code_names(
    mut v: PyObjectRef,
    field: &str,
) -> Result<Box<[String]>, crate::PyError> {
    if !unsafe { is_tuple(v) } {
        return Err(crate::PyError::type_error(format!(
            "{field} must be a tuple of strings"
        )));
    }
    let n = pyre_object::w_tuple_len(v);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut e = pyre_object::w_tuple_getitem(v, i as i64).unwrap_or_else(pyre_object::w_none);
        if !unsafe { pyre_object::is_str(e) } {
            return Err(crate::PyError::type_error(format!(
                "{field} must be a tuple of strings"
            )));
        }
        out.push(pyre_object::with_roots!(v, e => crate::baseobjspace::str_utf8_w(e))?.to_string());
    }
    Ok(out.into_boxed_slice())
}

/// A `bytes` `co_*` field (linetable / exceptiontable) as raw bytes.
unsafe fn read_code_bytes(v: PyObjectRef, field: &str) -> Result<Box<[u8]>, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(v) } {
        return Err(crate::PyError::type_error(format!(
            "{field} must be a bytes object"
        )));
    }
    Ok(unsafe { pyre_object::bytesobject::bytes_like_data(v) }
        .to_vec()
        .into_boxed_slice())
}

/// `co_code` bytes → the decoded `CodeUnits` instruction stream.  The byte
/// form is the `original_bytes` layout: one `(opcode, arg)` pair per unit.
unsafe fn read_code_units(
    v: PyObjectRef,
) -> Result<(crate::bytecode::CodeUnits, Option<Vec<u8>>), crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(v) } {
        return Err(crate::PyError::type_error("co_code must be a bytes object"));
    }
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(v) };
    decode_code_units(bytes)
        .map_err(|()| crate::PyError::value_error("co_code length must be a multiple of 2"))
}

/// Decode public `co_code` bytes into compiler-core's execution storage while
/// retaining an exact fallback for opcode values its enum cannot represent.
///
/// The marshal runtime bag calls this same boundary, so `CodeType`,
/// `code.replace` and marshal loading cannot disagree about which bytes are
/// deferred to dispatch as `Instruction::Reserved`.
pub(crate) fn decode_code_units(
    bytes: &[u8],
) -> Result<(crate::bytecode::CodeUnits, Option<Vec<u8>>), ()> {
    if !bytes.len().is_multiple_of(2) {
        return Err(());
    }
    let mut units = Vec::with_capacity(bytes.len() / 2);
    let mut preserve_raw = false;
    for pair in bytes.as_chunks::<2>().0 {
        let (op, arg) = match crate::bytecode::Instruction::try_from(pair[0]) {
            Ok(crate::bytecode::Instruction::Reserved) | Err(_) => {
                preserve_raw = true;
                // `CodeUnit` cannot carry an arbitrary opcode byte.  Keep the
                // exact public stream in `PyCode.co_code_bytes`, and carry the
                // invalid opcode in the Reserved placeholder's otherwise
                // meaningless arg so shared interpreter/JIT dispatch can
                // report CPython's `unknown opcode N` at execution time.
                (
                    crate::bytecode::Instruction::Reserved,
                    crate::bytecode::OpArgByte::from(pair[0]),
                )
            }
            Ok(op) => (op, crate::bytecode::OpArgByte::from(pair[1])),
        };
        units.push(crate::bytecode::CodeUnit::new(op, arg));
    }
    Ok((
        crate::bytecode::CodeUnits::from(units),
        preserve_raw.then(|| bytes.to_vec()),
    ))
}

/// A `tuple` `co_consts` field → the compiler `Constants` backing table.
///
/// PyPy stores the supplied wrapped objects directly in `co_consts_w`
/// (`pycode.py`).  Pyre's compiler table still needs one entry per wrapped
/// object so bytecode indices remain valid, but it is not the semantic owner:
/// values the compiler enum cannot represent use `None` only as an unobserved
/// shape placeholder.  `w_code_fill_consts_from_tuple` immediately installs
/// every supplied object in the authoritative wrapped slots, and LOAD_CONST
/// plus marshal both read those slots.
unsafe fn read_code_consts(
    v: PyObjectRef,
) -> Result<crate::bytecode::Constants<crate::bytecode::ConstantData>, crate::PyError> {
    if !unsafe { is_tuple(v) } {
        return Err(crate::PyError::type_error("co_consts must be a tuple"));
    }
    // Element conversion allocates (bigint digits, nested tuples) and can
    // collect. Pin the tuple and reload it before every getitem.
    let _roots = pyre_object::gc_roots::push_roots();
    let slot = pyre_object::gc_roots::pin_roots(&[v]);
    let n = pyre_object::w_tuple_len(pyre_object::gc_roots::shadow_stack_get(slot));
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e =
            pyre_object::w_tuple_getitem(pyre_object::gc_roots::shadow_stack_get(slot), i as i64)
                .unwrap_or_else(pyre_object::w_none);
        out.push(unsafe { obj_to_constant_data(e) }.unwrap_or(crate::bytecode::ConstantData::None));
    }
    Ok(out.into_iter().collect())
}

/// Convert a Python object into compiler `ConstantData` when that enum can
/// represent it.  Callers that need a literal compiler constant propagate the
/// `ValueError`; `read_code_consts` instead supplies a shape placeholder and
/// retains the arbitrary object in PyPy's authoritative `co_consts_w` slot.
pub(crate) unsafe fn obj_to_constant_data(
    obj: PyObjectRef,
) -> Result<crate::bytecode::ConstantData, crate::PyError> {
    use crate::bytecode::ConstantData;
    unsafe {
        if is_none(obj) {
            return Ok(ConstantData::None);
        }
        if is_ellipsis(obj) {
            return Ok(ConstantData::Ellipsis);
        }
        // bool is a subclass of int, so test it first.
        if is_bool(obj) {
            let value = crate::builtins::space_index_w(obj)? != 0;
            return Ok(ConstantData::Boolean { value });
        }
        if is_int_or_long(obj) {
            return Ok(ConstantData::Integer {
                value: crate::rbigint_to_compiler_bigint(&crate::builtins::obj_to_bigint(obj)),
            });
        }
        if is_float(obj) {
            return Ok(ConstantData::Float {
                value: pyre_object::w_float_get_value(obj),
            });
        }
        if pyre_object::is_complex(obj) {
            return Ok(ConstantData::Complex {
                value: num_complex::Complex64::new(
                    pyre_object::w_complex_get_real(obj),
                    pyre_object::w_complex_get_imag(obj),
                ),
            });
        }
        if pyre_object::is_str(obj) {
            return Ok(ConstantData::Str {
                value: pyre_object::w_str_get_wtf8(obj).to_owned(),
            });
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return Ok(ConstantData::Bytes {
                value: pyre_object::bytesobject::bytes_like_data(obj).to_vec(),
            });
        }
        if is_tuple(obj) {
            // Recursive conversion of an element can collect. Pin the tuple
            // and reload it before every getitem.
            let _roots = pyre_object::gc_roots::push_roots();
            let slot = pyre_object::gc_roots::pin_roots(&[obj]);
            let n = pyre_object::w_tuple_len(pyre_object::gc_roots::shadow_stack_get(slot));
            let mut elements = Vec::with_capacity(n);
            for i in 0..n {
                let e = pyre_object::w_tuple_getitem(
                    pyre_object::gc_roots::shadow_stack_get(slot),
                    i as i64,
                )
                .unwrap_or_else(pyre_object::w_none);
                elements.push(obj_to_constant_data(e)?);
            }
            return Ok(ConstantData::Tuple { elements });
        }
        if pyre_object::sliceobject::is_slice(obj) {
            let _roots = pyre_object::gc_roots::push_roots();
            let slot = pyre_object::gc_roots::pin_roots(&[obj]);
            return Ok(ConstantData::Slice {
                elements: Box::new([
                    obj_to_constant_data(pyre_object::sliceobject::w_slice_get_start(
                        pyre_object::gc_roots::shadow_stack_get(slot),
                    ))?,
                    obj_to_constant_data(pyre_object::sliceobject::w_slice_get_stop(
                        pyre_object::gc_roots::shadow_stack_get(slot),
                    ))?,
                    obj_to_constant_data(pyre_object::sliceobject::w_slice_get_step(
                        pyre_object::gc_roots::shadow_stack_get(slot),
                    ))?,
                ]),
            });
        }
        if pyre_object::setobject::is_frozenset(obj) {
            let items = pyre_object::w_set_items(obj);
            // `w_set_items` returns a Vec the collector cannot see. Pin every
            // item before converting any of them.
            let _roots = pyre_object::gc_roots::push_roots();
            let item_base = pyre_object::gc_roots::pin_roots(&items);
            let n = items.len();
            let mut elements = Vec::with_capacity(n);
            for i in 0..n {
                let item = pyre_object::gc_roots::shadow_stack_get(item_base + i);
                elements.push(obj_to_constant_data(item)?);
            }
            return Ok(ConstantData::Frozenset { elements });
        }
        // The inverse of `pyframe.rs`'s `pyobject_from_constant`, which realizes
        // this constant as a slice object: a subscript with only literal bounds
        // (`p[:0]`) folds to one, so a module that has any is unconvertible
        // without this arm.
        if pyre_object::is_slice(obj) {
            let _roots = pyre_object::gc_roots::push_roots();
            let slot = pyre_object::gc_roots::pin_roots(&[obj]);
            let start = obj_to_constant_data(pyre_object::w_slice_get_start(
                pyre_object::gc_roots::shadow_stack_get(slot),
            ))?;
            let stop = obj_to_constant_data(pyre_object::w_slice_get_stop(
                pyre_object::gc_roots::shadow_stack_get(slot),
            ))?;
            let step = obj_to_constant_data(pyre_object::w_slice_get_step(
                pyre_object::gc_roots::shadow_stack_get(slot),
            ))?;
            return Ok(ConstantData::Slice {
                elements: Box::new([start, stop, step]),
            });
        }
        if is_code(obj) {
            let ptr = w_code_get_ptr(obj) as *const crate::CodeObject;
            if !ptr.is_null() {
                return Ok(ConstantData::Code {
                    code: Box::new((*ptr).clone()),
                });
            }
        }
        Err(crate::PyError::value_error(
            "co_consts contains a value that is not a valid code constant",
        ))
    }
}

/// `pyopcode.py getconstant_w(index) -> co_consts_w[index]`.
///
/// # Safety
/// `w_code_obj` must be null or point to a valid `PyCode`.
pub unsafe fn w_code_const(w_code_obj: PyObjectRef, idx: usize) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.base();
    let w_code_obj = roots.pin_root(w_code_obj);
    let w_code = unsafe { &*(w_code_obj as *const PyCode) };
    // Guard `code_ptr` before dereferencing it — the same null/alignment check
    // the lazy-cache initializers use. A null/misaligned pointer means the
    // nested code is unreadable, so return PY_NULL and let the caller realize
    // the constant from its own code object.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    if w_code.code_ptr.is_null() || (w_code.code_ptr as i64) & align_mask != 0 {
        return pyre_object::pyobject::PY_NULL;
    }
    let code = unsafe { &*(w_code.code_ptr as *const crate::CodeObject) };
    let constants = crate::pyframe::code_constants(code);
    // closure-free, Option-pattern-free `constants.get(idx)` rewrite — keep the
    // bounds check a plain `lt + getitem` ahead of the variant destructure.
    if idx >= constants.len() {
        return pyre_object::pyobject::PY_NULL;
    }
    let table = w_code.co_consts_w;
    if table.is_null() {
        return crate::pyframe::pyobject_from_constant(&constants[idx]);
    }
    if idx >= unsafe { (&*table).len() } {
        return pyre_object::pyobject::PY_NULL;
    }
    // Pin the table across the realizing allocation. PyCode is old-gen, so
    // a shadow-stack visit of the wrapper does not scan `co_consts_w`; the
    // collector rewrites this pin into the shadow slot and the return value.
    // `getconstant_w` indexes the live `co_consts_w` array.
    let table_slot = pyre_object::gc_roots::shadow_stack_len();
    let table = roots.pin_root(table as pyre_object::PyObjectRef) as *mut FixedObjectArray;
    let existing = unsafe { (&*table).load_atomic(idx, std::sync::atomic::Ordering::Acquire) };
    if !existing.is_null() {
        return existing;
    }

    let realized = unsafe { realize_code_constant(code_slot, idx) };
    let realized_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(realized);
    let table = pyre_object::gc_roots::shadow_stack_get(table_slot) as *mut FixedObjectArray;
    // Barrier-then-CAS through the array slot. The barrier is not a
    // collection point (`remember_young_pointer`); it must run before
    // the publish so a concurrent minor already holding the table in
    // the remembered set still traces the new young child.
    let realized = roots.get(realized_slot);
    match unsafe {
        (*table).compare_exchange_ref(
            idx,
            pyre_object::pyobject::PY_NULL,
            realized,
            std::sync::atomic::Ordering::AcqRel,
            std::sync::atomic::Ordering::Acquire,
        )
    } {
        Ok(_) => realized,
        Err(winner) => winner,
    }
}

/// `pyopcode.py getname_w(index) -> self.getcode().co_names_w[index]`.
///
/// # Safety
/// `w_code_obj` must point to a `PyCode` whose `co_names_w` covers `idx`.
pub unsafe fn w_code_getname_w(w_code_obj: PyObjectRef, idx: usize) -> PyObjectRef {
    let table = unsafe { (*(w_code_obj as *const PyCode)).co_names_w };
    unsafe { (&*table)[idx] }
}

/// Whether `idx` names a slot of `w_code_obj`'s `co_names_w`.
///
/// False for a null object, an empty name array (unusable `code_ptr`), and
/// for [`crate::pyopcode::NO_NAMEINDEX`].
pub(crate) unsafe fn w_code_name_index_covered(w_code_obj: PyObjectRef, idx: usize) -> bool {
    if w_code_obj.is_null() {
        return false;
    }
    let table = unsafe { (*(w_code_obj as *const PyCode)).co_names_w };
    idx < unsafe { (*table).len() }
}

/// [`w_code_getname_w`] when `idx` is a real `co_names` slot; otherwise the
/// caller's own literal, interned.
///
/// A null or unaligned `code_ptr` holds a zero-length table, and
/// [`crate::pyopcode::NO_NAMEINDEX`] is not a slot (the implicit class-body
/// `__class__` store). Neither asks [`w_code_getname_w`] for a null name.
///
/// # Safety
/// `w_code_obj` must be null or point to a valid `PyCode`.
pub unsafe fn w_code_getname_w_or_new(
    w_code_obj: PyObjectRef,
    idx: usize,
    name: &str,
) -> PyObjectRef {
    if unsafe { w_code_name_index_covered(w_code_obj, idx) } {
        return unsafe { w_code_getname_w(w_code_obj, idx) };
    }
    pyre_object::unicodeobject::intern_str_value(name)
}

/// pypy/module/__pypy__/interp_magic.py:79
/// `func.getcode().hidden_applevel = True` — explicit setter for the
/// `__pypy__.hidden_applevel(func)` builtin marker, plus the
/// `_continuation.entrypoint_pycode.hidden_applevel = True`
/// hand-edit (interp_continuation.py:195).  PyPy mutates the field
/// directly; pyre wraps the raw write because the field is private
/// to this module.
///
/// Not every `Code` reaching here is a `PyCode`: `interp_w(Function, ...)`
/// admits `BuiltinFunction(Function)` (`function.py`), whose `getcode()` is a
/// `BuiltinCode`.  Upstream's assignment is a no-op for that one — `gateway.py
/// BuiltinCode` already declares `hidden_applevel = True` at class level, and
/// `eval.py Code` lists the field in `_immutable_fields_`.  pyre's write is a
/// raw store at `PyCode`'s own offset, so it must first establish that the
/// object has that layout; anything else keeps the value it already reports.
///
/// # Safety
/// `obj` must point to a valid object.
#[inline]
pub unsafe fn w_code_set_hidden_applevel(obj: PyObjectRef, hidden_applevel: bool) {
    if obj.is_null() || !unsafe { is_code(obj) } {
        return;
    }
    unsafe {
        (*(obj as *mut PyCode)).hidden_applevel = hidden_applevel;
    }
}

/// `assemble.py make_code` stamps `compile_info.hidden_applevel` on every
/// code object one compilation produces. Marshal rebuilds those wrappers
/// with the default `False`, so a cached unit sets the flag on the graph
/// `box_code_object` already published in `co_consts_w`.
///
/// # Safety
/// `w_code` must be a `PyCode`.
pub unsafe fn set_hidden_applevel_unit(w_code: PyObjectRef) {
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if !pycode.co_consts_w.is_null() {
        for &nested in unsafe { (&*pycode.co_consts_w).as_slice() } {
            if !nested.is_null() && unsafe { is_code(nested) } {
                unsafe { set_hidden_applevel_unit(nested) };
            }
        }
    }
    unsafe { w_code_set_hidden_applevel(w_code, true) };
}

/// Extract the leaked host `CodeObject` address from a known PyCode.
///
/// `co_code` is immutable; a constant `PyCode` folds this to the body
/// pointer so decode can fold `code_unit_at` as well. The return is
/// `*const CodeObject` (`PyCode.code_ptr`): `history.getkind` of that
/// raw `Ptr` is `"int"`, matching `code_instructions_len`'s `FUNC.ARGS`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub unsafe fn w_code_get_ptr(obj: PyObjectRef) -> *const crate::CodeObject {
    unsafe { (*(obj as *const PyCode)).code_ptr }
}

/// `Constants<ConstantData>` is a single-field tuple wrapping `Box<[ConstantData]>`.
/// `DerefMut` is gone; recover a mutable slice from that layout so nested
/// constants keep their addresses. rustpython-compiler-core exposes only
/// `as_tuple()` (`&C::Tuple`); there is no `&mut` accessor.
const _: () = {
    assert!(
        std::mem::size_of::<Constants<crate::bytecode::ConstantData>>()
            == std::mem::size_of::<Box<[crate::bytecode::ConstantData]>>()
    );
    assert!(
        std::mem::align_of::<Constants<crate::bytecode::ConstantData>>()
            == std::mem::align_of::<Box<[crate::bytecode::ConstantData]>>()
    );
};

fn constants_as_mut_slice(
    constants: &mut Constants<crate::bytecode::ConstantData>,
) -> &mut [crate::bytecode::ConstantData] {
    debug_assert_eq!(
        std::mem::size_of_val(constants),
        std::mem::size_of::<Box<[crate::bytecode::ConstantData]>>()
    );
    debug_assert_eq!(
        std::mem::align_of_val(constants),
        std::mem::align_of::<Box<[crate::bytecode::ConstantData]>>()
    );
    // SAFETY: `Constants<C>` is `struct Constants<C>(C::Tuple)` and
    // `ConstantData::Tuple` is `Box<[ConstantData]>`. Size and align match
    // the compile-time asserts above; nested constant addresses stay put.
    let tuple = unsafe {
        &mut *std::ptr::from_mut(constants).cast::<Box<[crate::bytecode::ConstantData]>>()
    };
    tuple.as_mut()
}

/// Resolve compiler-emitted GET_ITER yield-from mode 1 to the coro-check
/// modes `fixup_get_iter` writes, put BUILD_LIST_FROM_ARG back in front of
/// GET_ITER, and walk nested code constants.
pub fn fixup_code_get_iter(code: &mut crate::CodeObject) {
    code.instructions.fixup_get_iter(code.flags);
    reorder_build_list_from_arg(code);
    for constant in constants_as_mut_slice(&mut code.constants) {
        fixup_constant_get_iter(constant);
    }
}

/// Temporary patch until the codegen emits BUILD_LIST_FROM_ARG before
/// GET_ITER on the non-inlined list-comprehension path, as the inlined path
/// already does. The pinned compiler emits `GET_ITER` (plus its CACHE units)
/// then `BUILD_LIST_FROM_ARG`; GET_ITER pushes the iterator and an index slot,
/// so BUILD_LIST_FROM_ARG (pop one value, push the list, push the value back)
/// would take the NULL index slot. Rotate each such pair to
/// `BUILD_LIST_FROM_ARG; GET_ITER (+ caches)`, which leaves
/// `[list, iterator, index]` on the stack, and move the per-unit locations
/// with it.
fn reorder_build_list_from_arg(code: &mut crate::CodeObject) {
    use crate::bytecode::{CodeUnit, Instruction};
    let len = code.instructions.len();
    let ptr = code.instructions.units_ptr() as *mut CodeUnit;
    let mut i = 0usize;
    while i < len {
        let unit = code.instructions[i];
        let last = i + 1 + unit.op.cache_entries();
        if matches!(unit.op, Instruction::GetIter { .. })
            && last < len
            && matches!(code.instructions[last].op, Instruction::BuildListFromArg)
        {
            // SAFETY: `i..=last` lies inside the `len` units the code object owns.
            unsafe { std::slice::from_raw_parts_mut(ptr.add(i), last - i + 1) }.rotate_right(1);
            if code.locations.len() == len {
                code.locations[i..=last].rotate_right(1);
            }
        }
        i += 1 + code.instructions[i].op.cache_entries();
    }
}

fn fixup_constant_get_iter(constant: &mut crate::bytecode::ConstantData) {
    match constant {
        crate::bytecode::ConstantData::Code { code } => fixup_code_get_iter(code),
        crate::bytecode::ConstantData::Tuple { elements }
        | crate::bytecode::ConstantData::Frozenset { elements } => {
            for nested in elements {
                fixup_constant_get_iter(nested);
            }
        }
        crate::bytecode::ConstantData::Slice { elements } => {
            for nested in elements.iter_mut() {
                fixup_constant_get_iter(nested);
            }
        }
        _ => {}
    }
}

/// `importing.py update_code_filenames`: set `source_path` on `code` and,
/// recursively, on every nested code constant whose filename still matches the
/// root's *original* name (leaving unrelated inlined filenames untouched).
///
/// Nested constants are wrapped in place by
/// [`box_code_constant_in_place`], and a wrapper published before this runs
/// reads the new name out of the object it already points at — which is what
/// `update_code_filenames` mutating already-wrapped nested `PyCode` constants
/// amounts to.
fn fix_code_filenames(code: &mut crate::CodeObject, oldname: &str, newname: &str) {
    code.source_path = newname.to_owned();
    for constant in constants_as_mut_slice(&mut code.constants) {
        if let crate::bytecode::ConstantData::Code { code: nested } = constant
            && nested.source_path == oldname
        {
            fix_code_filenames(nested, oldname, newname);
        }
    }
}

/// `_imp._fix_co_filename(code, path)` (importing.py:158 fix_co_filename):
/// replace `co_filename` on the code object in place, recursing through its
/// nested code constants.  Only valid because the code object is owned
/// (`Box::into_raw`), single-threaded, and not executing when import machinery
/// calls this right after compilation.
///
/// # Safety
/// `w_code` must point to a valid `PyCode` whose body is not currently running.
pub unsafe fn fix_co_filename(w_code: PyObjectRef, newname: &[u8]) {
    unsafe { fix_co_filename_oldname(w_code, newname, None) }
}

/// `importing.py update_code_filenames` / `fix_co_filename`: rewrite this
/// code's filename first, then walk nested constants by index, reloading the
/// wrapper and `co_consts_w` after each nested call.
unsafe fn fix_co_filename_oldname(mut w_code: PyObjectRef, newname: &[u8], oldname: Option<&[u8]>) {
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.pin_roots(&[w_code]);
    w_code = roots.get(code_slot);
    let code_ptr = unsafe { w_code_get_ptr(w_code) } as *mut crate::CodeObject;
    if code_ptr.is_null() {
        return;
    }
    // `importing.py update_code_filenames` stores `pathname` on every nested
    // code whose filename still equals the root's. When the root already
    // spells that path, the walk would write the same bytes back.
    {
        let pycode = unsafe { &*(w_code as *const PyCode) };
        let already = if pycode.filename_bytes.is_null() {
            unsafe { &*(pycode.code_ptr as *const crate::CodeObject) }
                .source_path
                .as_bytes()
                == newname
        } else {
            unsafe { &*pycode.filename_bytes }.as_slice() == newname
        };
        if already {
            return;
        }
    }
    let old_filename = {
        let current = unsafe { code_filename_bytes(roots.get(code_slot)) };
        w_code = roots.get(code_slot);
        if let Some(old) = oldname
            && current.as_slice() != old
        {
            return;
        }
        current
    };
    let old_source_path = unsafe { (*code_ptr).source_path.clone() };
    w_code = roots.get(code_slot);
    let filename = match std::str::from_utf8(newname) {
        Ok(newname_str) => {
            unsafe { (*code_ptr).source_path = newname_str.to_owned() };
            None
        }
        Err(_) => Some(newname.to_vec()),
    };
    unsafe { set_filename_bytes(roots.get(code_slot), filename) };
    w_code = roots.get(code_slot);
    unsafe {
        (*(roots.get(code_slot) as *mut PyCode)).filename_inherits_to_nested =
            std::str::from_utf8(newname).is_err();
    }
    let mut i = 0usize;
    loop {
        let mut nested = {
            let pycode = unsafe { &*(roots.get(code_slot) as *const PyCode) };
            if pycode.co_consts_w.is_null() {
                break;
            }
            let table = unsafe { &*pycode.co_consts_w };
            if i >= table.len() {
                break;
            }
            table[i]
        };
        i += 1;
        if nested.is_null() || !unsafe { is_code(nested) } {
            continue;
        }
        let nested_scope = pyre_object::gc_roots::push_roots();
        let nested_slot = nested_scope.pin_roots(&[nested]);
        nested = nested_scope.get(nested_slot);
        if unsafe { code_filename_bytes(nested_scope.get(nested_slot)) } == old_filename {
            unsafe {
                fix_co_filename_oldname(
                    nested_scope.get(nested_slot),
                    newname,
                    Some(old_filename.as_slice()),
                )
            };
        }
        nested = nested_scope.get(nested_slot);
        let _ = nested;
        w_code = roots.get(code_slot);
    }
    if let Ok(newname_str) = std::str::from_utf8(newname) {
        let code_ptr = unsafe { w_code_get_ptr(roots.get(code_slot)) } as *mut crate::CodeObject;
        if !code_ptr.is_null() {
            unsafe { fix_code_filenames(&mut *code_ptr, &old_source_path, newname_str) };
        }
    }
    w_code = roots.get(code_slot);
    let _ = w_code;
}

/// Cached [`crate::pyframe::npure_cellvars`] for the code wrapper `obj`,
/// or `None` for a null/stub wrapper (sentinel `u32::MAX`) so the caller
/// falls back to recomputation.
///
/// # Safety
/// `obj` must be null or point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_npure_cellvars(obj: PyObjectRef) -> Option<usize> {
    if obj.is_null() {
        return None;
    }
    match unsafe { (*(obj as *const PyCode)).npure_cellvars } {
        u32::MAX => None,
        n => Some(n as usize),
    }
}

/// PyPy: `PyCode.hidden_applevel` (`pycode.py`). Reads the field
/// initialised by `w_code_new`.  `pyframe.py:521-522
/// hide(self): return self.pycode.hidden_applevel` is the sole caller
/// in the canonical interpreter; pyre routes through this accessor
/// from `pyframe.rs::PyFrame::hide`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_hidden_applevel(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe { (*(obj as *const PyCode)).hidden_applevel }
}

/// PyPy: `PyCode.w_globals` — the globals dict OBJECT. The JIT
/// codewriter/bridge read this to fold globals lookups without an off-GC
/// proxy.
///
/// `get_w_globals` promotes the code object first; with that constant
/// argument this residual folds the same way `w_code_get_ptr` does.
#[inline]
#[majit_macros::elidable_cannot_raise]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_code_get_w_globals(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::PY_NULL;
    }
    // Paired with the publication in `w_code_frame_stores_global`.
    unsafe {
        std::sync::atomic::AtomicPtr::from_ptr(std::ptr::addr_of_mut!(
            (*(obj as *mut PyCode)).w_globals
        ))
        .load(std::sync::atomic::Ordering::Acquire)
    }
}

/// `quasiimmut.py get_current_qmut_instance` for `pycode.py` `"w_globals?"`.
///
/// Resolved while the read is recorded, so a later store in the same trace
/// already has a watcher to sweep, and the compiled loop registers on the
/// same instance.
///
/// # Safety
/// `obj` must point to a live [`PyCode`], or be null.
pub unsafe fn w_code_current_w_globals_qmut(
    obj: PyObjectRef,
) -> Option<std::sync::Arc<pyre_object::quasiimmut::QuasiImmut>> {
    if obj.is_null() {
        return None;
    }
    Some(unsafe {
        (*(obj as *const PyCode))
            .w_globals_watchers
            .get_current_qmut_instance()
    })
}

/// PyPy: `PyCode.w_globals = w_globals`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_code_set_w_globals(obj: PyObjectRef, w_globals: PyObjectRef) {
    if obj.is_null() {
        return;
    }
    // A bootstrap code slot is reached only by the prebuilt root walk, which
    // clean minor collections may skip; record the store before making it.
    let published = publish_code_slot_store_rooting(obj, &[w_globals]);
    let obj = published.owner();
    let w_globals = published.get(0);
    unsafe {
        // `rclass.py hook_setfield` runs `jit_force_quasi_immutable` ahead of
        // every store to a `?` field (`pycode.py` `"w_globals?"`). The hook
        // does not look at the new value, so storing the value already there
        // invalidates too. The sweep and the store share the watcher lock:
        // an `is_installed` test outside it lets a recorder publish a watcher
        // for the old pointer after the test and before the store.
        let code = obj as *mut PyCode;
        pyre_object::quasiimmut::publish_quasi_immut_ptr(
            &(*code).w_globals_watchers,
            std::ptr::addr_of_mut!((*code).w_globals),
            w_globals,
        );
    }
    if !w_globals.is_null() {
        let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr as *const () };
        register_live_code_wrapper(code_ptr, obj);
        register_w_globals_stamped_code(obj);
    }
}

/// PyPy: `PyCode.frame_stores_global(w_globals)`.
///
/// A slot that already holds a namespace is only compared by identity: a
/// frame answered `false` keeps reading the `w_globals` it was handed, which
/// is that same pointer, so nothing is read through the loaded value and a
/// `Relaxed` load answers it. Only the first store, which has to be
/// indivisible from its read, goes through [`w_code_publish_frame_global`].
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_code_frame_stores_global(obj: PyObjectRef, w_globals: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let published = unsafe {
        std::sync::atomic::AtomicPtr::from_ptr(std::ptr::addr_of_mut!(
            (*(obj as *mut PyCode)).w_globals
        ))
        .load(std::sync::atomic::Ordering::Relaxed)
    };
    if published.is_null() {
        return unsafe { w_code_publish_frame_global(obj, w_globals) };
    }
    // The wrapper may have moved since its first globals stamp. The live frame
    // supplying `obj` is authoritative, so keep the raw-code compatibility
    // registry synchronized even on the cache-hit path.
    w_code_register_live_wrapper(obj);
    !std::ptr::eq(published, w_globals)
}

/// Refresh [`register_live_code_wrapper`] for the code object `obj`.
#[majit_macros::dont_look_inside]
fn w_code_register_live_wrapper(obj: PyObjectRef) {
    register_live_code_wrapper(
        unsafe { (*(obj as *const PyCode)).code_ptr as *const () },
        obj,
    );
}

/// The first `frame_stores_global` store into a code object's `w_globals`.
#[majit_macros::dont_look_inside]
unsafe fn w_code_publish_frame_global(obj: PyObjectRef, w_globals: PyObjectRef) -> bool {
    let roots = pyre_object::gc_roots::push_roots();
    let root_base = roots.publish(&[obj, w_globals]);
    roots.normalize(root_base, 2);
    let obj = roots.get(root_base);
    // `pycode.py frame_stores_global` reads the slot and stores into it under
    // the GIL, which makes the pair indivisible. Pyre is free-threaded, and a
    // `false` answer is what makes a frame take its globals from the code
    // object: two threads first running one code object in different globals
    // both read the null, and an unsynchronized store would let both answer
    // `false`, leaving the loser's frame reading the winner's namespace.
    // Publish by compare-exchange and answer against whichever pointer won.
    let code = obj as *mut PyCode;
    let slot = unsafe {
        std::sync::atomic::AtomicPtr::from_ptr(std::ptr::addr_of_mut!((*code).w_globals))
    };
    let mut published = slot.load(std::sync::atomic::Ordering::Acquire);
    if published.is_null() {
        // Prebuilt-family store (see `w_code_set_w_globals`): the barrier runs
        // before the publication, so a collection that observes the new child
        // through this slot finds the slot already remembered. A lost race
        // leaves only a conservative barrier behind.
        let rooted =
            publish_code_slot_store_rooting(roots.get(root_base), &[roots.get(root_base + 1)]);
        let obj = rooted.owner();
        let code = obj as *mut PyCode;
        let w_globals = rooted.get(0);
        // `rclass.py hook_setfield` ahead of the store to `pycode.py`
        // `"w_globals?"`. The null check, the sweep and the store share the
        // watcher lock. A lost race must not sweep the winner's watcher.
        match unsafe {
            pyre_object::quasiimmut::publish_quasi_immut_cas(
                &(*code).w_globals_watchers,
                std::ptr::addr_of_mut!((*code).w_globals),
                pyre_object::PY_NULL,
                w_globals,
            )
        } {
            Ok(()) => {
                register_live_code_wrapper(unsafe { (*code).code_ptr as *const () }, obj);
                register_w_globals_stamped_code(obj);
                return false;
            }
            Err(winner) => published = winner,
        }
    }
    let obj = roots.get(root_base);
    let w_globals = roots.get(root_base + 1);
    register_live_code_wrapper(
        unsafe { (*(obj as *const PyCode)).code_ptr as *const () },
        obj,
    );
    !std::ptr::eq(published, w_globals)
}

/// The state of a code object's `co_positions` rows once its own `locations`
/// array has been released.
///
/// `Deferred` carries the first line number the rows must be decoded against.
/// `CodeObject.first_line_number` cannot stand in for it: it is an
/// `Option<OneIndexed>`, which cannot spell the zero and negative values
/// `CodeType(...)` accepts and `co_firstlineno_raw` preserves.
#[derive(Clone, Copy)]
enum CodeLocations {
    Deferred(i32),
    Decoded(&'static [(SourceLocation, SourceLocation)]),
}

/// Rows released from `CodeObject.locations`, keyed by code object address.
///
/// `locations` is not serialized. `offset2lineno` walks the line table when a
/// line is asked for, so marshal and `code.replace` leave the expanded array
/// empty. [`code_locations`] rebuilds it on the first reader — the
/// realize-once treatment `co_consts_w` and `co_names_w` already get. A reader
/// is a traceback, a debugger line jump, or the `co_positions` / `co_lines`
/// getters.
///
/// Decoded rows are leaked because the `CodeObject` describing them is itself
/// never released: `pycode_destructor` frees the side tables and leaves
/// `code_ptr` standing, so an entry can never outlive its key's validity and
/// never has to be revisited.
///
/// Process-global and keyed by `usize` for `LIVE_CODE_WRAPPERS`' reasons: the
/// rows belong to the shared code object, and a raw `PyObjectRef` is not
/// `Send`.
static CODE_LOCATIONS: std::sync::OnceLock<
    parking_lot::Mutex<std::collections::HashMap<usize, CodeLocations>>,
> = std::sync::OnceLock::new();

fn code_locations_cache()
-> &'static parking_lot::Mutex<std::collections::HashMap<usize, CodeLocations>> {
    CODE_LOCATIONS.get_or_init(|| parking_lot::Mutex::new(std::collections::HashMap::new()))
}

/// Release a freshly constructed code object's expanded `locations` array,
/// recording the first line number [`code_locations`] must decode it back
/// against.
///
/// Called before the wrapper is published, so no reader can be holding the
/// array that is dropped here, and a later call for the same code object (the
/// `co_firstlineno_raw` stamp that follows `CodeType(...)` and `code.replace`)
/// simply corrects the recorded line number.
fn release_code_locations(code_ptr: *mut crate::CodeObject, firstlineno_raw: i32) {
    // Nested constants are wrapped in place, so two threads realizing one
    // `co_consts_w` slot reach the same `CodeObject` before either wrapper is
    // published. The lock is taken and the vacant entry claimed before the
    // pointer is dereferenced, so only the thread that owns the record ever
    // forms a `&mut` to the object: the array is read and written under this
    // lock alone. A second wrapper over an already-released object must keep
    // the record the first made — re-deferring would abandon rows a reader has
    // since decoded.
    let mut cache = code_locations_cache().lock();
    let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(code_ptr as usize) else {
        return;
    };
    let code = unsafe { &mut *code_ptr };
    // No instructions means there is no line table to rebuild. An empty
    // `locations` array is the deferred state: the marshal reader and
    // `code.replace` leave it empty, and `offset2lineno` expands on demand.
    if code.instructions.is_empty() {
        return;
    }
    entry.insert(CodeLocations::Deferred(firstlineno_raw));
    if !code.locations.is_empty() {
        code.locations = Vec::new().into_boxed_slice();
    }
}

/// Correct the first line number a released array is decoded against, leaving a
/// code object that still holds its own array alone.
fn record_deferred_locations_firstlineno(code_ptr: *mut crate::CodeObject, firstlineno_raw: i32) {
    let mut cache = code_locations_cache().lock();
    if let Some(slot @ CodeLocations::Deferred(_)) = cache.get_mut(&(code_ptr as usize)) {
        *slot = CodeLocations::Deferred(firstlineno_raw);
    }
}

/// The `co_positions` rows of `code`, decoding them out of `linetable` on the
/// first reader when [`release_code_locations`] has taken the array away.
///
/// Returns the code object's own array untouched when it still holds one, so a
/// code object built outside [`w_code_new`] reads exactly as it always did.
///
/// An empty array on a code object that has instructions always means the rows
/// were released: every construction path fills `locations` through
/// `linetable_to_locations`, which returns one row per instruction. Decoding is
/// therefore the right answer whether or not the map still knows the object.
pub fn code_locations(code: &crate::CodeObject) -> &[(SourceLocation, SourceLocation)] {
    if !code.locations.is_empty() || code.instructions.is_empty() {
        return &code.locations;
    }
    let key = code as *const crate::CodeObject as usize;
    let mut cache = code_locations_cache().lock();
    let firstlineno_raw = match cache.get(&key) {
        Some(CodeLocations::Decoded(rows)) => return rows,
        Some(CodeLocations::Deferred(firstlineno_raw)) => *firstlineno_raw,
        // A copy of a released code object: `w_code_fill_consts_from_tuple`
        // serializes a wrapped constant back into `ConstantData` by copying the
        // `CodeObject`, so the empty array travels to an address this map has
        // never seen. `linetable` is the serialized form and travels with it,
        // which is why it — not the map — is what makes the rows recoverable;
        // the record only exists to carry a first line number
        // `Option<OneIndexed>` cannot spell.
        None => code
            .first_line_number
            .map(|line| line.get() as i32)
            .unwrap_or(1),
    };
    let rows: &'static [(SourceLocation, SourceLocation)] =
        Box::leak(rustpython_compiler_core::marshal::linetable_to_locations(
            &code.linetable,
            firstlineno_raw,
            code.instructions.len(),
        ));
    cache.insert(key, CodeLocations::Decoded(rows));
    rows
}

/// `PyCode_Addr2Line` — the source line the byte offset `addrq` sits on, or
/// `-1` when the offset names no line.
///
/// ```c
/// if (addrq < 0) {
///     return co->co_firstlineno;
/// }
/// ...
/// return _PyCode_CheckLineNumber(addrq, &bounds);
/// ```
///
/// `location.py offset2lineno(c, stopat)` is the same resolver upstream, down
/// to the byte-offset convention: it halves `stopat` before walking the line
/// table.  `pyframe::offset2lineno` carries the name but takes an instruction
/// index, so this is the one a `tb_lasti` reaches.
///
/// `addrq` counts bytes, two per instruction — the same unit
/// `traceback._get_code_position` works in app-level.
///
/// EVERY negative offset names the code object's first line, which is the
/// answer for a frame that has not run an instruction yet.  Upstream tests
/// `stopat == -1` alone, so `-2` reaches `_offset2lineno(..., -1)`, walks a
/// zero-length range and answers `-1`.  Measured with a traceback rebuilt at
/// each offset:
///
/// | `tb_lasti` | 3.14.2 | PyPy 7.3.20 |
/// |---|---|---|
/// | `-1` | `co_firstlineno` | `co_firstlineno` |
/// | `-2`, `-3` | `co_firstlineno` | `-1` |
///
/// The `addrq < 0` arm above is 3.14's, and `traceback_lineno_sentinel.py`
/// pins it.
///
/// The answer is signed and is **not** narrowed to "a line or nothing":
/// `_PyCode_CheckLineNumber`'s `-1` and a genuinely negative line are one
/// value upstream too, and the two readers that render it as `None` —
/// `tb_lineno_get` and `frame_getlineno` — test `< 0` themselves.  `repr` of
/// a frame does not, and prints whatever came back, so a code object whose
/// table spells line `-8` has to reach `frame_repr` as `-8`.  `-1` therefore
/// also covers an offset past the last range AND an offset inside a
/// `NO_LOCATION` range — the compiler-generated cleanup a `with` or a
/// `try`/`finally` leaves at the end of a code object, which `co_positions`
/// reports with a `None` line.
///
/// The walk goes through [`PyCodeAddressRange`], not [`code_locations`]: the
/// expanded array holds a `SourceLocation` per instruction and its `line` is a
/// `OneIndexed`, so a `NO_LOCATION` range comes back carrying whatever line
/// preceded it.  Reading `linetable` directly also takes no lock, where the
/// array is behind a process-global mutex once `w_code_new` releases it.
///
/// # Safety
/// `w_code` must be a live `PyCode`.
///
/// This body is `offset2lineno` and is looked inside, so a trace folds the
/// negative-offset arm to `co_firstlineno`.  The table walk is
/// [`_offset2lineno`], the residual call.
pub unsafe fn w_code_addr2line(w_code: PyObjectRef, addrq: i64) -> i32 {
    unsafe {
        if (*(w_code as *const PyCode)).code_ptr.is_null() {
            return -1;
        }
        if addrq < 0 {
            return w_code_firstlineno_raw(w_code);
        }
        _offset2lineno(w_code, addrq)
    }
}

/// `location.py _offset2lineno`: the linetable walk behind
/// [`w_code_addr2line`], for `addrq >= 0`.  It loops, so `look_inside_graph`
/// refuses it upstream; `#[dont_look_inside]` keeps it the same residual call.
/// The per-code memo of resolved offsets lives here, beside the walk it saves.
///
/// # Safety
/// `w_code` must be a live `PyCode` whose `code_ptr` is not null.
#[majit_macros::dont_look_inside]
unsafe fn _offset2lineno(w_code: PyObjectRef, addrq: i64) -> i32 {
    unsafe {
        let code = &*(w_code_get_ptr(w_code) as *const crate::CodeObject);
        let firstlineno = w_code_firstlineno_raw(w_code);
        // An offset outside `i32` cannot be spelled in the packed pair.
        let Ok(key) = i32::try_from(addrq) else {
            return code_addr2line(code, firstlineno, addrq);
        };
        let memo =
            &(*(w_code as *const PyCode)).addr2line_memo[key as usize & (ADDR2LINE_MEMO_WAYS - 1)];
        let packed = memo.load(std::sync::atomic::Ordering::Relaxed);
        if (packed >> 32) as i32 == key {
            return packed as i32;
        }
        let line = code_addr2line(code, firstlineno, addrq);
        memo.store(
            ((key as i64) << 32) | (line as u32 as i64),
            std::sync::atomic::Ordering::Relaxed,
        );
        line
    }
}

/// `PyCode_Addr2Line`'s body, split from the object it reads its two inputs
/// from so a bare `CodeObject` and a first line number can be walked directly.
///
/// `firstlineno` is `co->co_firstlineno`, which `_PyCode_InitAddressRange`
/// seeds `computed_line` with whatever its sign.  It has to come from the
/// `PyCode` wrapper's `co_firstlineno_raw` stamp: `CodeObject
/// .first_line_number` is an `Option<OneIndexed>`, `None` for every value
/// `OneIndexed` cannot spell, and `CodeType(...)` accepts the negative ones —
/// only `code.replace(co_firstlineno=...)` rejects them.  Reconstructing the
/// integer from the option therefore reads `0` for a code object whose
/// `co_firstlineno` is `-1`, and decodes the table against a different first
/// line than `co_lines()` and `co_positions()` do.
fn code_addr2line(code: &crate::CodeObject, firstlineno: i32, addrq: i64) -> i32 {
    if addrq < 0 {
        return firstlineno;
    }
    let lasti = i32::try_from(addrq).unwrap_or(i32::MAX);
    let mut bounds = PyCodeAddressRange::new(&code.linetable, firstlineno);
    bounds.check_line_number(lasti)
}

/// Specialization rewrites the opcode byte in place (`RESUME` becomes
/// `RESUME_CHECK`), so every read asks about the family.
fn base_op(code: &crate::CodeObject, index: usize) -> Option<crate::bytecode::Instruction> {
    let op = code.instructions.get(index)?.op;
    Some(op.to_base().unwrap_or(op))
}

/// `code->_co_firsttraceable` — the index of the first `RESUME` that is not a
/// generator expression's `RESUME_AT_GEN_EXPR_START`, which is where a frame
/// that has not run an instruction is reported to be.
///
/// `init_code` (`Objects/codeobject.c`) scans for it once and stores it; this
/// walks instead, which costs the prologue's length because it stops at that
/// `RESUME` — within a handful of units of the start for every code object the
/// compiler produces.  Everything ahead of it is what the compiler inserts:
/// `COPY_FREE_VARS`, `MAKE_CELL`, a generator expression's
/// `RESUME RESUME_AT_GEN_EXPR_START; LOAD_FAST .0; GET_ITER`, and a
/// generator's `RETURN_GENERATOR` / `POP_TOP` pair.  A code object with no
/// such `RESUME` at all leaves the index past the end, so it starts no line
/// anywhere — what `init_code`'s scan does with one too.
pub fn first_traceable_index(code: &crate::CodeObject) -> usize {
    use crate::bytecode::Instruction;
    use crate::bytecode::oparg::{ResumeContext, ResumeLocation};

    let mut index = 0;
    while let Some(op) = base_op(code, index) {
        if matches!(op, Instruction::Resume { .. })
            && u32::from(u8::from(code.instructions[index].arg)) & ResumeContext::LOCATION_MASK
                != u32::from(ResumeLocation::AtGenExprStart.as_u8())
        {
            break;
        }
        index += 1;
    }
    index
}

/// The index the frame of a generator that has not been resumed rests on.
///
/// `RETURN_GENERATOR` is what builds the generator object, so by the time a
/// caller holds it that instruction has run and the frame it left behind sits
/// on the next one.  The index is not a constant: a closure generator's
/// `COPY_FREE_VARS` precedes `RETURN_GENERATOR` and shifts it along.
///
/// `0` for a code object with no `RETURN_GENERATOR`, which is every code
/// object that is not a generator, coroutine or async generator; a frame of
/// one starts at index 0.  A found index is at least 1.
///
/// The scan reads only opcode families, which specialization keeps, so the
/// answer is a function of the code object: `@jit.elidable`, like
/// [`crate::pyframe::npure_cellvars`].  `PyFrame::next_instr` calls it on
/// every dispatch of a frame that has not run an instruction yet.
#[majit_macros::elidable_cannot_raise]
pub fn after_return_generator_index(code: &crate::CodeObject) -> usize {
    use crate::bytecode::Instruction;

    let mut index = 0;
    while let Some(op) = base_op(code, index) {
        if matches!(op, Instruction::ReturnGenerator) {
            return index + 1;
        }
        index += 1;
    }
    0
}

/// Whether the instruction at `pc` may be reported to a trace function as the
/// start of a source line.
///
/// Both rules come from `initialize_lines` (`Python/instrumentation.c`), which
/// decides once per code object which instructions carry `INSTRUMENTED_LINE`:
///
/// * `if (i < code->_co_firsttraceable)` — the frame prologue starts no line.
///   `_co_firsttraceable` is the index of the first `RESUME` (`init_code`,
///   `Objects/codeobject.c`), so this covers everything the compiler inserts
///   ahead of it: `COPY_FREE_VARS`, `MAKE_CELL`, and a generator's
///   `RETURN_GENERATOR` / `POP_TOP` pair, which carries the `def` line.
/// * a switch excluding `END_ASYNC_FOR`, `END_FOR`, `END_SEND`, `RESUME` and
///   `POP_ITER`.  `END_FOR` and `END_SEND` are skipped by the `FOR_ITER` /
///   `SEND` ahead of them; the other three carry no line event of their own
///   even though the line table gives them a line.
/// * cache units that follow an instruction.  `initialize_lines` walks
///   `_PyInstruction_GetLength` and stamps `for (j = 1; j < length; j++)`
///   so a `CACHE` never carries `INSTRUMENTED_LINE`.  `RESUME` now has
///   `cache_entries=1`, and that unit inherits the `def` line; tracing it
///   would put a line event on the `def` line at every call.
///
/// [`crate::PyFrame::get_lineno_for_pc_tracing`] needs both because the
/// prologue and the `RESUME` that ends it are all stamped with the `def` line,
/// so tracing any of them as a line start puts a line event on the `def` line
/// at every call.  `pycode.py` cannot answer this: its 3.11 bytecode has none
/// of these opcodes, and where its own `_get_lineno_for_pc_tracing` has no line
/// for a pc it answers `-1`, which is the sentinel `run_trace_func` already
/// guards on.
pub fn instruction_can_start_a_line(code: &crate::CodeObject, pc: usize) -> bool {
    use crate::bytecode::Instruction;

    // `i < code->_co_firsttraceable`.
    if pc < first_traceable_index(code) {
        return false;
    }

    let Some(op) = base_op(code, pc) else {
        return false;
    };
    !matches!(
        op,
        Instruction::Cache
            | Instruction::EndAsyncFor
            | Instruction::EndFor
            | Instruction::EndSend
            | Instruction::PopIter
            | Instruction::Resume { .. }
    )
}

/// Registry mapping a raw CodeObject pointer (`PyCode.code_ptr`) to the
/// live, globals-stamped `PyCode` wrapper. Populated where a frame stamps
/// the wrapper's `w_globals` — the only point both the raw pointer and the
/// live wrapper are in hand — and consumed by the JIT to recover the live
/// wrapper (and hence its `w_globals`) from a raw code pointer it already
/// holds, so the JIT need not carry the wrapper identity as a separate
/// `w_code` courier. First-write-wins, mirroring the first-store-wins
/// `PyCode.w_globals` semantics in `w_code_frame_stores_global`.
/// `MetaInterpStaticData.jitcodes` roots the wrappers it retains and forwards
/// this mapping during every minor collection; an executing frame also
/// refreshes the mapping from its live `pycode` field. The collector destructor
/// removes every other mapping before its pointer can dangle.
///
/// Process-global, not per-thread: `pycode.py` keeps `w_globals` on the
/// shared `PyCode` instance, and a code object stamped on one thread must be
/// recoverable from every thread that later runs it — a thread-local map made
/// the JIT's `recover_inline_callee_globals` answer `PY_NULL` there and
/// decline the inline. Keys and values are `usize` because a raw
/// `PyObjectRef` is not `Send`, as in `interp_sre`'s pattern registry.
///
/// Keyed by an address, so it is hashed the way `lldict.py _hash` hashes one
/// (`majit_gc::address_dict::AddressHasher`). Rust's default `SipHash` was 4.3%
/// of a call-heavy interpreted loop on its own, because every frame creation
/// refreshes its own entry.
static LIVE_CODE_WRAPPERS: std::sync::OnceLock<
    parking_lot::Mutex<majit_gc::address_dict::AddressMap<usize>>,
> = std::sync::OnceLock::new();

fn live_code_wrappers() -> &'static parking_lot::Mutex<majit_gc::address_dict::AddressMap<usize>> {
    LIVE_CODE_WRAPPERS
        .get_or_init(|| parking_lot::Mutex::new(majit_gc::address_dict::AddressMap::default()))
}

/// Record `wrapper` as the live wrapper for `code_ptr`. No-op on null inputs.
///
/// The frame/function field supplying `wrapper` is the authoritative GC edge.
/// Refreshing an existing entry is required now that `PyCode` follows
/// `pycode.py class PyCode(W_Root)` and can move between calls.
pub fn register_live_code_wrapper(code_ptr: *const (), wrapper: PyObjectRef) {
    if code_ptr.is_null() || wrapper.is_null() {
        return;
    }
    let wrapper = wrapper as usize;
    let mut wrappers = live_code_wrappers().lock();
    match wrappers.get_mut(&(code_ptr as usize)) {
        // The refresh exists for a wrapper that moved, and an unmoved one is
        // the common case: every frame this code object runs comes back here.
        // Leaving the slot alone spares the store and, on a table at its load
        // factor, the rehash that inserting over an existing key still risks.
        Some(live) if *live == wrapper => {}
        Some(live) => *live = wrapper,
        None => {
            wrappers.insert(code_ptr as usize, wrapper);
        }
    }
}

/// Latch assembler overflow onto the live wrapper so the per-frame shape
/// gate agrees with `CallControl.graph_jit_shapes` without a map probe.
pub fn store_wrapper_jit_shape_overflow(code_ptr: *const ()) {
    let w = live_code_wrapper(code_ptr);
    if w.is_null() {
        return;
    }
    unsafe {
        (*(w as *const PyCode)).jit_shape.store(
            JIT_SHAPE_STORED_CONST_ENCODING_OVERFLOW,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// Recover the live wrapper previously registered for `code_ptr`, or `PY_NULL`
/// if none has been stamped.
pub fn live_code_wrapper(code_ptr: *const ()) -> PyObjectRef {
    if code_ptr.is_null() {
        return PY_NULL;
    }
    let mut wrappers = live_code_wrappers().lock();
    let Some(wrapper) = wrappers.get_mut(&(code_ptr as usize)) else {
        return PY_NULL;
    };
    // A collection root walk may have installed a forwarding pointer before
    // this side table itself is visited. Follow it eagerly so every reader
    // sees the live address and write the answer back for later readers.
    *wrapper = pyre_object::gc_hook::try_gc_current_object_address(
        *wrapper as pyre_object::gc_hook::GCREF,
    ) as usize;
    *wrapper as PyObjectRef
}

/// Forward one JitCode-retained live-wrapper entry in place.
///
/// Called only from the collector's per-mutator root-area walk while that
/// mutator is quiesced. `MetaInterpStaticData.jitcodes` is the strong owner,
/// matching `warmspot.py self.metainterp_sd.jitcodes`; the address-keyed map is
/// only pyre's raw-CodeObject compatibility seam.
pub fn walk_live_code_wrapper(code_ptr: *const (), forward: &mut dyn FnMut(&mut PyObjectRef)) {
    if code_ptr.is_null() {
        return;
    }
    let mut wrappers = live_code_wrappers().lock();
    let Some(wrapper) = wrappers.get_mut(&(code_ptr as usize)) else {
        return;
    };
    let mut root = *wrapper as PyObjectRef;
    forward(&mut root);
    *wrapper = root as usize;
}

/// One `box_code_object` allocation graph and what still reads it.
///
/// `pycode.py` needs no equivalent.  A nested code object is an ordinary GC
/// object upstream, so the collector reaches each one through `co_consts_w`
/// and reclaims it on its own.  Pyre's compiler hands over a single
/// recursively-owned `CodeObject` graph instead, and
/// [`box_code_constant_in_place`] publishes wrappers that point *into* it, so
/// nothing in the graph may be released until every wrapper reached through it
/// is gone -- a closure outliving the module that defined it keeps its own
/// `__code__` inside its parent's `constants` table.
struct CodeUnit {
    /// Live `PyCode` wrappers whose `code_ptr` lands in this graph: the
    /// top-level one plus one per realized nested code constant.
    wrappers: usize,
    /// A jitcode keeps a raw body address (`PyJitCodePayload::code_ptr`) and
    /// dereferences it long after the wrappers are gone, so once one is
    /// written the graph has to stay readable for the process lifetime.
    pinned: bool,
    /// Every `code_ptr` registered against this graph, so [`retire_code_unit`]
    /// can drop their entries in the address-keyed side tables together with
    /// the bodies those entries describe.
    members: Vec<usize>,
}

#[derive(Default)]
struct CodeUnitRegistry {
    /// Keyed by the owning `Box::into_raw` address.
    units: std::collections::HashMap<usize, CodeUnit>,
    /// Every published `code_ptr` to the graph that owns it.
    owner_of: std::collections::HashMap<usize, usize>,
}

/// Both halves live under one lock: [`retire_code_unit`] walks from a unit to
/// its members, and [`pin_code_unit`] walks the other way, so a lock per map
/// would be two orders over the same pair.
static CODE_UNITS: std::sync::OnceLock<std::sync::Mutex<CodeUnitRegistry>> =
    std::sync::OnceLock::new();

fn code_units() -> &'static std::sync::Mutex<CodeUnitRegistry> {
    CODE_UNITS.get_or_init(|| std::sync::Mutex::new(CodeUnitRegistry::default()))
}

/// Record a wrapper published over `code_ptr` inside the graph `owner` owns.
///
/// `owner` is zero for the constructors whose contract is that the body is
/// never released ([`w_code_new`]'s prebuilt and gateway callers), and those
/// register nothing at all.
fn attach_code_unit_wrapper(owner: usize, code_ptr: *const ()) {
    if owner == 0 || code_ptr.is_null() {
        return;
    }
    let mut registry = code_units()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let registry = &mut *registry;
    let unit = registry.units.entry(owner).or_insert_with(|| CodeUnit {
        wrappers: 0,
        pinned: false,
        members: Vec::new(),
    });
    unit.wrappers += 1;
    if registry.owner_of.insert(code_ptr as usize, owner).is_none() {
        unit.members.push(code_ptr as usize);
    }
}

/// Drop one wrapper's claim on its graph, retiring the graph with the last one.
fn detach_code_unit_wrapper(code_ptr: *const ()) {
    if code_ptr.is_null() {
        return;
    }
    let mut guard = code_units()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let registry = &mut *guard;
    let Some(&owner) = registry.owner_of.get(&(code_ptr as usize)) else {
        return;
    };
    let Some(unit) = registry.units.get_mut(&owner) else {
        return;
    };
    unit.wrappers -= 1;
    if unit.wrappers > 0 || unit.pinned {
        return;
    }
    let unit = registry
        .units
        .remove(&owner)
        .expect("unit was just borrowed");
    for member in &unit.members {
        registry.owner_of.remove(member);
    }
    // The graph is out of both maps, so no other caller can reach it, and the
    // retirement below takes a second lock and frees arbitrary payloads.  It
    // runs with this one released so neither is an order over the other.
    drop(guard);
    retire_code_unit(owner as *mut crate::CodeObject, &unit.members);
}

/// The `box_code_object` allocation `code_ptr`'s body belongs to, or zero when
/// no graph claims it.
fn code_unit_owner(code_ptr: *const ()) -> usize {
    if code_ptr.is_null() {
        return 0;
    }
    code_units()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .owner_of
        .get(&(code_ptr as usize))
        .copied()
        .unwrap_or(0)
}

/// Keep the graph `code_ptr` belongs to readable for the process lifetime.
///
/// The JIT stores raw body addresses in its jitcode table and dereferences
/// them from trace-side readers that hold no wrapper, so a code object it has
/// written a jitcode for cannot take part in the retirement above.  Callers
/// hold a live wrapper for `code_ptr`, which is what keeps the graph in the
/// registry for this to find.
pub fn pin_code_unit(code_ptr: *const ()) {
    if code_ptr.is_null() {
        return;
    }
    let mut registry = code_units()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let registry = &mut *registry;
    let Some(&owner) = registry.owner_of.get(&(code_ptr as usize)) else {
        return;
    };
    if let Some(unit) = registry.units.get_mut(&owner) {
        unit.pinned = true;
    }
}

/// Drop everything a retired graph's bodies own, keeping every allocation the
/// graph is addressed by.
///
/// The bodies are what the memory is in -- the instruction units, the constant
/// payloads, the name tables and the two line/exception tables -- and nothing
/// can read them once the last wrapper is gone.  The `CodeObject` shells and
/// the `constants` spines that hold them stay allocated on purpose: a raw
/// `CodeObject` address is the key of every side table in the tree (the JIT's
/// jitcode map and its per-code shape cache, [`CODE_LOCATIONS`], the JIT's
/// green keys), and none of those is reachable from here, so an address handed
/// back to the allocator would answer another code object's lookups with this
/// one's record.  Holding the shells is what keeps an address unique for the
/// process lifetime, which is the property those tables were written against.
fn retire_code_unit(owner: *mut crate::CodeObject, members: &[usize]) {
    {
        let mut cache = code_locations_cache().lock();
        for member in members {
            // A `Decoded` record's rows stay leaked: `code_locations` hands
            // them out as `&'static`, and this runs from the collector, which
            // can interrupt a reader still holding one.  Dropping the record
            // is enough -- a retired body has no instructions, so
            // `code_locations` answers from the empty array and never consults
            // the map for it again.
            cache.remove(member);
        }
    }
    // SAFETY: `owner` is the `Box::into_raw` address `box_code_object`
    // registered, no wrapper reaches any body in the graph any more, and the
    // registry entry has been removed under the lock, so this runs once.
    release_code_object_body(unsafe { &mut *owner });
}

/// Release one body's payloads and recurse through the code constants nested
/// in it, which are the graph's other members.
fn release_code_object_body(code: &mut crate::CodeObject) {
    for constant in constants_as_mut_slice(&mut code.constants) {
        release_constant_payload(constant);
    }
    code.instructions = crate::bytecode::CodeUnits::from(Vec::new());
    code.locations = Vec::new().into_boxed_slice();
    code.names = Vec::new().into_boxed_slice();
    code.varnames = Vec::new().into_boxed_slice();
    code.cellvars = Vec::new().into_boxed_slice();
    code.freevars = Vec::new().into_boxed_slice();
    code.localspluskinds = Vec::new().into_boxed_slice();
    code.linetable = Vec::new().into_boxed_slice();
    code.exceptiontable = Vec::new().into_boxed_slice();
    code.source_path = String::new();
    code.obj_name = String::new();
    code.qualname = String::new();
}

/// Free a retired constant's payload, keeping the `Box<CodeObject>` a nested
/// code constant is addressed by.
///
/// A code object can sit inside a container constant as well as directly in
/// the table (`ConstantData::Tuple`'s elements), so the container variants
/// recurse rather than being replaced wholesale.
fn release_constant_payload(constant: &mut crate::bytecode::ConstantData) {
    use crate::bytecode::ConstantData;
    match constant {
        ConstantData::Code { code } => release_code_object_body(code),
        ConstantData::Tuple { elements } | ConstantData::Frozenset { elements } => {
            for element in elements.iter_mut() {
                release_constant_payload(element);
            }
        }
        ConstantData::Slice { elements } => {
            for element in elements.iter_mut() {
                release_constant_payload(element);
            }
        }
        // Scalars own no allocation worth the discriminant change; the rest
        // are the strings, bytes and bignums the table is mostly made of.
        ConstantData::Integer { .. } | ConstantData::Str { .. } | ConstantData::Bytes { .. } => {
            *constant = ConstantData::None
        }
        ConstantData::Float { .. }
        | ConstantData::Complex { .. }
        | ConstantData::Boolean { .. }
        | ConstantData::None
        | ConstantData::Ellipsis => {}
    }
}

/// Collector destructor for a managed `PyCode` wrapper.
///
/// PyPy lets the GC reclaim the `PyCode` and its list-valued cache fields
/// together.  Pyre's three cache vectors are raw Rust allocations, so release
/// them here and retire the compatibility registries that used to assume every
/// wrapper was immortal.  `code_ptr` deliberately remains allocated: compiled
/// JitCodes currently retain that raw compiler-body address independently of
/// the wrapper, and freeing it belongs to the later removal of that courier.
///
/// # Safety
/// `obj_addr` must be a collector-owned `PyCode` payload and this function must
/// run at most once for it.
pub unsafe fn pycode_destructor(obj_addr: usize) {
    let code = unsafe { &mut *(obj_addr as *mut PyCode) };
    // The collector runs no `Drop`, so the hidden `mutate_w_globals`
    // (`pycode.py` `"w_globals?"`) would otherwise keep its instance and its
    // out-of-line lock. Releasing one strong count leaves an in-flight
    // compile's clone alive. Same release `property_destructor` performs for
    // `w_fget?`.
    unsafe { code.w_globals_watchers.reclaim() };
    let wrapper = obj_addr as PyObjectRef;
    {
        let mut wrappers = live_code_wrappers().lock();
        if wrappers.get(&(code.code_ptr as usize)).copied() == Some(wrapper as usize) {
            wrappers.remove(&(code.code_ptr as usize));
        }
    }
    w_globals_stamped_codes().lock().remove(&obj_addr);
    mapdict_method_cache_codes().lock().remove(&obj_addr);
    if !code.cell_families.is_null() {
        // Only the slot table. The families themselves stay allocated: cells
        // created from this code may still be alive in a surviving closure,
        // and each holds an untraced `Cell.family` pointer into them.
        //
        // That is why freeing them here is wrong, not why never freeing them is
        // right. One family is minted per cellvar per code-object construction
        // and none is ever reclaimed, so a process that keeps compiling grows
        // without bound -- measured at ~201 bytes retained per cellvar per
        // compile. `pycode.py` `_initialize` and `nestedscope.py` `Cell.__init__`
        // hold the family through two ordinary references and let the collector
        // take it once the table and every surviving cell are gone; reaching
        // that shape here means tracing both edges, which is deferred rather
        // than done. Immortality is also what currently makes the untraced word
        // safe for the recorder's `quasi_immut_descr` read, so the two move
        // together.
        //
        // The code object this hangs off is already leaked whole and on purpose
        // (`box_code_object`), at a far larger byte rate, so this does not
        // change what a workload can do today.
        drop(unsafe { Box::from_raw(code.cell_families) });
        code.cell_families = std::ptr::null_mut();
    }
    if !code.globals_caches.is_null() {
        drop(unsafe { Box::from_raw(code.globals_caches) });
        code.globals_caches = std::ptr::null_mut();
    }
    if !code.mapdict_caches.is_null() {
        drop(unsafe { Box::from_raw(code.mapdict_caches) });
        code.mapdict_caches = std::ptr::null_mut();
    }
    let loop_header_info = code
        .loop_header_info
        .swap(std::ptr::null_mut(), std::sync::atomic::Ordering::Acquire);
    if !loop_header_info.is_null() {
        drop(unsafe { Box::from_raw(loop_header_info) });
    }
    if !code.co_consts_w.is_null() {
        // Only the no-hook `alloc_mro_block_gc` fallback is freed here; a
        // GC-owned table is the collector's.
        unsafe { pyre_object::dealloc_mro_block(code.co_consts_w) };
        code.co_consts_w = std::ptr::null_mut();
    }
    if !code.co_names_w.is_null() {
        unsafe { pyre_object::dealloc_mro_block(code.co_names_w) };
        code.co_names_w = std::ptr::null_mut();
    }
    if !code.filename_bytes.is_null() {
        drop(unsafe { Box::from_raw(code.filename_bytes) });
        code.filename_bytes = std::ptr::null_mut();
    }
    if !code.co_code_bytes.is_null() {
        drop(unsafe { Box::from_raw(code.co_code_bytes) });
        code.co_code_bytes = std::ptr::null_mut();
    }
    // Last, so every side table this wrapper owns is already gone when the
    // graph's own bodies are released.
    detach_code_unit_wrapper(code.code_ptr as *const ());
}

/// pycode.py `_compute_flatcall`.
///
/// Returns FLATPYCALL | co_argcount for simple user functions (no *args,
/// **kwargs, keyword-only args). Returns HOPELESS otherwise.
fn compute_flatcall(code: &crate::CodeObject) -> u16 {
    use crate::CodeFlags;
    use crate::gateway::{BuiltinCodeFlags, HOPELESS};
    if code
        .flags
        .intersects(CodeFlags::VARARGS | CodeFlags::VARKEYWORDS)
    {
        return HOPELESS;
    }
    if code.kwonlyarg_count > 0 {
        return HOPELESS;
    }
    if code.arg_count > 0xff {
        return HOPELESS;
    }
    // pycode.py:234 — disqualify if any arg is also a cellvar.
    // Pyre's CodeObject exposes cellvars; check for overlap.
    let argcount = code.arg_count as usize;
    if !code.cellvars.is_empty() && argcount > 0 {
        for cellname in &code.cellvars {
            for j in 0..argcount {
                if j < code.varnames.len() && *cellname == code.varnames[j] {
                    return HOPELESS;
                }
            }
        }
    }
    BuiltinCodeFlags::FLATPYCALL.bits() | (code.arg_count as u16)
}

/// eval.py:16-23 — read `fast_natural_arity` from a PyCode.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe { (*(obj as *const PyCode)).fast_natural_arity & !YIELDS_INSIDE_TRY_BIT }
}

/// Unified accessor: read `fast_natural_arity` from any code object
/// (BuiltinCode or PyCode).
///
/// # Safety
/// `obj` must point to a valid code object (either type).
#[inline]
pub unsafe fn code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe {
        if crate::gateway::is_builtin_code(obj) {
            crate::gateway::builtin_code_get_fast_natural_arity(obj)
        } else {
            w_code_get_fast_natural_arity(obj)
        }
    }
}

/// pycode.py `PyCode.lookup_exceptiontable`.
///
/// Search the wrapped code object's exception table for a handler
/// covering `instr_offset` (byte offset into `co_code`).  Returns
/// `Some((target, depth, lasti))` with byte-offset `target` when found.
///
/// `pycode.py PyCode.lookup_exceptiontable` is `@jit.elidable`.
///
/// The residual ABI is one register. Pack the handler as
/// `(target << 32) | (depth << 1) | lasti`; `-1` is no handler.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
#[majit_macros::elidable]
pub unsafe extern "C" fn w_code_lookup_exceptiontable(obj: PyObjectRef, instr_offset: u32) -> i64 {
    if obj.is_null() {
        return pack_exceptiontable_lookup(None);
    }
    let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
    if code_ptr.is_null() {
        return pack_exceptiontable_lookup(None);
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    pack_exceptiontable_lookup(crate::pycode::lookup_exceptiontable(
        &code.exceptiontable,
        instr_offset,
    ))
}

/// Residual ABI is one i64 per slot on every backend, including wasm32.
pub extern "C" fn w_code_lookup_exceptiontable_jit_abi(obj: PyObjectRef, instr_offset: i64) -> i64 {
    unsafe { w_code_lookup_exceptiontable(obj, instr_offset as u32) }
}

/// No handler: `-1`. A hit cannot collide because `target`/`depth` are
/// non-negative bytecode/stack quantities.
pub const EXCEPTIONTABLE_LOOKUP_NONE: i64 = -1;

pub fn pack_exceptiontable_lookup(hit: Option<(u32, u32, bool)>) -> i64 {
    match hit {
        None => EXCEPTIONTABLE_LOOKUP_NONE,
        Some((target, depth, lasti)) => {
            debug_assert!(depth < (1u32 << 31));
            // Build in `u64` so a target with the high bit set does not
            // look like the `-1` sentinel after the shift.
            (((target as u64) << 32) | ((depth as u64) << 1) | u64::from(lasti)) as i64
        }
    }
}

pub fn unpack_exceptiontable_lookup(packed: i64) -> Option<(u32, u32, bool)> {
    if packed == EXCEPTIONTABLE_LOOKUP_NONE {
        None
    } else {
        let bits = packed as u64;
        Some(((bits >> 32) as u32, (bits >> 1) as u32, bits & 1 != 0))
    }
}

/// pycode.py `self.co_exceptiontable = exceptiontable` — copy the
/// varint-packed table bytes out of the wrapped `CodeObject`.
///
/// The bytes are owned by the inner `CodeObject` (`Box<[u8]>` field), so
/// returning a reference would tie the lifetime to the obj's heap
/// allocation.  Callers that need to hand the bytes to Python (where
/// they get copied into a `W_BytesObject`) take the owned `Vec<u8>`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_exceptiontable(obj: PyObjectRef) -> Vec<u8> {
    if obj.is_null() {
        return Vec::new();
    }
    let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
    if code_ptr.is_null() {
        return Vec::new();
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    code.exceptiontable.to_vec()
}

/// The `cell_families[...]` entry (`pycode.py` `PyCode._initialize`) for the
/// cellvar occupying localsplus
/// slot `slot` (`pyframe.py` `PyFrame.initialize_frame_scopes`).
///
/// Falls back to `nestedscope.py DUMMY_FAMILY` — never null — when the
/// slot names no cellvar or the code carries no family table, so a cell is
/// always constructible and one built without a family simply never folds.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_cell_family(obj: PyObjectRef, slot: usize) -> *const CellFamily {
    if obj.is_null() {
        return pyre_object::nestedscope::dummy_family();
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.cell_families.is_null() {
        return pyre_object::nestedscope::dummy_family();
    }
    match unsafe { &*code.cell_families }.get(slot) {
        Some(family) if !family.is_null() => *family,
        _ => pyre_object::nestedscope::dummy_family(),
    }
}

/// `celldict.py:292 cache_wref = pycode._globals_caches[nameindex]` —
/// read slot `nameindex` and upgrade the weakref to a strong
/// `Arc<Mutex<GlobalCache>>` (returning `None` when the slot is
/// unset, the weak target is gone, or `code_ptr` is invalid).
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<std::sync::Arc<parking_lot::Mutex<pyre_object::celldict::GlobalCache>>> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return None;
    }
    let vec = unsafe { &*code.globals_caches }.lock();
    vec.get(nameindex)
        .and_then(|slot| slot.as_ref())
        .and_then(|w| w.upgrade())
}

/// `celldict.py:321/353 pycode._globals_caches[nameindex] = cache.ref`
/// — store `Arc::downgrade(cache)` in slot `nameindex`.  No-op when
/// `code_ptr` is invalid or `nameindex` is out of range.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    cache: &std::sync::Arc<parking_lot::Mutex<pyre_object::celldict::GlobalCache>>,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return;
    }
    let mut vec = unsafe { &*code.globals_caches }.lock();
    if let Some(slot) = vec.get_mut(nameindex) {
        *slot = Some(std::sync::Arc::downgrade(cache));
    }
}

/// Number of `_globals_caches` slots — equals `len(co_names_w)` at
/// construction time.  Returns 0 for code objects built from null
/// or unaligned `code_ptr`.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return 0;
    }
    unsafe { (*code.globals_caches).lock().len() }
}

/// `mapdict.py/1546/1575 entry = pycode._mapdict_caches[nameindex]` — read
/// slot `nameindex`, returning `None` (PyPy `INVALID_CACHE_ENTRY`) when the slot
/// is unset, out of range, or `code_ptr` is invalid.  The entry is `Copy`, so a
/// value is returned (no aliasing of the slot).
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<crate::objspace::std::mapdict::MapdictCacheEntry> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return None;
    }
    let vec = unsafe { &*code.mapdict_caches };
    vec.get(nameindex).copied().flatten()
}

/// `mapdict.py pycode._mapdict_caches[nameindex] = entry` — store the
/// filled entry in slot `nameindex`.  No-op when `code_ptr` is invalid or
/// `nameindex` is out of range.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    mut entry: crate::objspace::std::mapdict::MapdictCacheEntry,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return;
    }
    let vec = unsafe { &mut *code.mapdict_caches };
    if let Some(slot) = vec.get_mut(nameindex) {
        // The LOAD_METHOD fill (mapdict.py:1474) stores a movable
        // `w_method` reference; register this code object so
        // `walk_mapdict_method_cache_gc` forwards the slot.
        if !entry.w_method.is_null() {
            if !pyre_object::gc_hook::try_gc_owns_object(obj as pyre_object::gc_hook::GCREF) {
                mapdict_method_cache_codes().lock().insert(obj as usize);
            }
            // The slot is reached only by `walk_mapdict_method_cache_gc`,
            // skipped on clean minors; record it before the store.
            let published = publish_code_slot_store_rooting(obj, &[entry.w_method]);
            entry.w_method = published.get(0);
        }
        *slot = Some(entry);
    }
}

/// Code objects whose `_mapdict_caches` hold (or once held) a filled
/// `w_method` slot.  In PyPy `CacheEntry.w_method` (mapdict.py)
/// is traced through the GC-managed `PyCode`; a managed wrapper's custom
/// trace reaches the slot the same way, so only a bootstrap wrapper minted
/// before the collector exists enters this registry and the extra-root
/// walker forwards it from here (same family as `walk_method_cache_gc`).
static MAPDICT_METHOD_CACHE_CODES: std::sync::OnceLock<
    parking_lot::Mutex<std::collections::HashSet<usize>>,
> = std::sync::OnceLock::new();

fn mapdict_method_cache_codes() -> &'static parking_lot::Mutex<std::collections::HashSet<usize>> {
    MAPDICT_METHOD_CACHE_CODES
        .get_or_init(|| parking_lot::Mutex::new(std::collections::HashSet::new()))
}

/// Code objects whose `w_globals` has been stamped
/// (`pycode.py frame_stores_global`).  `w_globals` is a permanent
/// strong field: the first globals object a code object runs in is kept
/// for the code object's lifetime and never replaced.  Upstream that
/// field is traced through the GC-managed `PyCode`, and a managed wrapper's
/// custom trace does the same.  A bootstrap wrapper minted before the
/// collector exists is reached only by the opportunistic
/// `walk_raw_code_roots` calls on `frame.pycode` and `func.code`, which miss
/// any stamped code object off the frame chain and not held in a walked frame
/// slot at collection time; that leaves a pre-move nursery address in
/// `w_globals` once its dict is promoted, and the next call through that code
/// object forwards a dangling pointer.  This registry makes the slot a root of
/// its own for that family, as [`MAPDICT_METHOD_CACHE_CODES`] does.
static W_GLOBALS_STAMPED_CODES: std::sync::OnceLock<
    parking_lot::Mutex<std::collections::HashSet<usize>>,
> = std::sync::OnceLock::new();

fn w_globals_stamped_codes() -> &'static parking_lot::Mutex<std::collections::HashSet<usize>> {
    W_GLOBALS_STAMPED_CODES
        .get_or_init(|| parking_lot::Mutex::new(std::collections::HashSet::new()))
}

/// Record `obj` as a code object holding a stamped `w_globals` slot.
fn register_w_globals_stamped_code(obj: PyObjectRef) {
    // A managed PyCode's own custom trace owns this edge.  Only bootstrap
    // wrappers outside the collector need the compatibility registry; making
    // every managed code's globals an independent root would turn
    // `code -> globals -> function/generator -> code` cycles immortal.
    if pyre_object::gc_hook::try_gc_owns_object(obj as pyre_object::gc_hook::GCREF) {
        return;
    }
    w_globals_stamped_codes().lock().insert(obj as usize);
}

/// Forward the stamped `w_globals` slot of every registered code object.
///
/// A process-global walker, not a per-mutator root area: a code object is
/// immortal and `w_globals` is first-store-wins, so the slot outlives whichever
/// thread happened to stamp it.  Owning the root per-thread meant
/// `unregister_mutator` dropped that thread's area at thread exit while the
/// code object stayed live and callable, and the slot then went unforwarded.
/// Same ownership argument as `interp_sre`'s pattern registry.
///
/// Reached only from the collector's root walk, where every mutator is at a
/// safepoint — that is what makes handing out `&mut code.w_globals` sound.
#[majit_macros::dont_look_inside]
pub fn walk_w_globals_stamped_code_roots(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    let codes = w_globals_stamped_codes().lock();
    for &code in codes.iter() {
        let code = unsafe { &mut *(code as *mut PyCode) };
        if code.w_globals.is_null() {
            continue;
        }
        forward(&mut code.w_globals);
    }
}

/// Forward every filled `entry.w_method` slot during collection — the
/// faithful equivalent of the GC tracing PyPy's `CacheEntry.w_method`
/// (mapdict.py) gets through its GC-managed holder.  The cached
/// map/attr node pointers are immortal interned nodes and the
/// `version_tag` is a `u64`, so `w_method` is the entry's only movable
/// reference.
///
/// Process-global for the same reason as
/// [`walk_w_globals_stamped_code_roots`]: the holder is immortal and outlives
/// the thread that filled the slot.
#[majit_macros::dont_look_inside]
pub fn walk_mapdict_method_cache_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    let codes = mapdict_method_cache_codes().lock();
    for &code in codes.iter() {
        let code = unsafe { &*(code as *const PyCode) };
        if code.mapdict_caches.is_null() {
            continue;
        }
        let vec = unsafe { &mut *code.mapdict_caches };
        for slot in vec.iter_mut() {
            if let Some(entry) = slot.as_mut()
                && !entry.w_method.is_null()
            {
                forward(&mut entry.w_method);
            }
        }
    }
}

/// Number of `_mapdict_caches` slots — equals `len(co_names_w)` at construction
/// time.  Returns 0 for code objects built from null or unaligned `code_ptr`.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return 0;
    }
    unsafe { (*code.mapdict_caches).len() }
}

/// Check if an object is a code object.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CODE_TYPE) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_exec;

    #[test]
    fn exceptiontable_lookup_pack_roundtrip() {
        assert_eq!(
            unpack_exceptiontable_lookup(pack_exceptiontable_lookup(None)),
            None
        );
        let hit = Some((100, 3, true));
        assert_eq!(
            unpack_exceptiontable_lookup(pack_exceptiontable_lookup(hit)),
            hit
        );
        let zero = Some((0, 0, false));
        assert_eq!(
            unpack_exceptiontable_lookup(pack_exceptiontable_lookup(zero)),
            zero
        );
        let high = Some((0x8000_0000, 4, false));
        assert_eq!(
            unpack_exceptiontable_lookup(pack_exceptiontable_lookup(high)),
            high
        );
        assert_eq!(pack_exceptiontable_lookup(None), EXCEPTIONTABLE_LOOKUP_NONE);
    }

    /// Build a minimal varint-encoded exception table from `(start, length,
    /// target, depth, lasti)` tuples, mirroring the encoding produced by
    /// `assemble.py::_encode_varint`. Values are passed as word offsets
    /// (the on-disk unit), not byte offsets.
    fn encode_table(entries: &[(u32, u32, u32, u32, bool)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (start, length, target, depth, lasti) in entries.iter().copied() {
            push_varint(&mut out, start, true);
            push_varint(&mut out, length, false);
            push_varint(&mut out, target, false);
            push_varint(&mut out, (depth << 1) | (lasti as u32), false);
        }
        out
    }

    fn push_varint(out: &mut Vec<u8>, mut value: u32, entry_start: bool) {
        let mut chunks = [0u8; 6];
        let mut n = 0;
        loop {
            chunks[n] = (value & 63) as u8;
            n += 1;
            value >>= 6;
            if value == 0 {
                break;
            }
        }
        for j in (0..n).rev() {
            let mut byte = chunks[j];
            if j != 0 {
                byte |= 0x40;
            }
            if j == n - 1 && entry_start {
                byte |= 0x80;
            }
            out.push(byte);
        }
    }

    /// The two tables disagree on byte order: `co_exceptiontable` puts the
    /// most significant 6-bit group first (`parse_varint`), `co_linetable`
    /// the least significant one (`write_varint`). 70 therefore encodes
    /// differently under each, and handing either reader the other's bytes
    /// yields 385 — the failure a shared decoder would produce on every
    /// multi-byte delta.
    #[test]
    fn the_two_varint_tables_use_opposite_byte_order() {
        let exception_bytes = [0x41u8, 0x06];
        let location_bytes = [0x46u8, 0x01];

        assert_eq!(decode_varint(&exception_bytes, 0), (70, 2));
        assert_eq!(LineTableReader::new(&location_bytes).read_varint(), 70);

        assert_eq!(decode_varint(&location_bytes, 0), (385, 2));
        assert_eq!(LineTableReader::new(&exception_bytes).read_varint(), 385);
    }

    /// `code.replace(co_linetable=...)` accepts arbitrary bytes, so the
    /// continuation chain can be longer than a u32 holds. Shifting by the full
    /// width panics in a debug build, so the groups past it are dropped.
    #[test]
    fn a_location_varint_chain_past_the_word_width_does_not_panic() {
        let overlong = [0x7fu8, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00];
        let mut expected = 0x3fu32;
        for shift in [6u32, 12, 18, 24, 30] {
            expected |= 0x3fu32 << shift;
        }
        assert_eq!(LineTableReader::new(&overlong).read_varint(), expected);
    }

    #[test]
    fn empty_table_returns_none() {
        assert_eq!(lookup_exceptiontable(&[], 0), None);
    }

    #[test]
    fn lookup_returns_byte_offsets() {
        // entry: word offsets start=4 (byte 8), length=10 (byte 20),
        // target=20 (byte 40), depth=2, lasti=false
        let table = encode_table(&[(4, 10, 20, 2, false)]);
        assert_eq!(lookup_exceptiontable(&table, 8), Some((40, 2, false)));
        assert_eq!(lookup_exceptiontable(&table, 27), Some((40, 2, false)));
        assert_eq!(lookup_exceptiontable(&table, 28), None);
        assert_eq!(lookup_exceptiontable(&table, 7), None);
    }

    #[test]
    fn last_matching_wins() {
        // Two overlapping ranges; outer first, inner second
        // (CPython emission order).
        let table = encode_table(&[(0, 10, 20, 1, false), (3, 4, 30, 3, true)]);
        assert_eq!(lookup_exceptiontable(&table, 2), Some((40, 1, false)));
        // PC 8 (byte) is covered by both. PyPy returns the later entry.
        assert_eq!(lookup_exceptiontable(&table, 8), Some((60, 3, true)));
        assert_eq!(lookup_exceptiontable(&table, 14), Some((40, 1, false)));
    }

    #[test]
    fn lasti_low_bit() {
        let table = encode_table(&[(0, 2, 10, 5, true)]);
        let (target, depth, lasti) = lookup_exceptiontable(&table, 0).unwrap();
        assert_eq!((target, depth, lasti), (20, 5, true));
    }

    #[test]
    fn iter_matches_lookup_count() {
        let table = encode_table(&[
            (0, 4, 8, 1, false),
            (10, 6, 20, 2, true),
            (30, 2, 40, 0, false),
        ]);
        let entries: Vec<_> = decode_exceptiontable(&table).collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1].start, 20);
        assert_eq!(entries[1].end, 32);
        assert_eq!(entries[1].target, 40);
        assert_eq!(entries[1].depth, 2);
        assert!(entries[1].lasti);
    }

    #[test]
    fn early_break_when_start_past_offset() {
        let table = encode_table(&[(0, 2, 10, 1, false), (100, 2, 200, 2, false)]);
        assert_eq!(lookup_exceptiontable(&table, 50), None);
    }

    #[test]
    fn live_code_wrapper_refreshes_to_latest_live_owner() {
        let code = 0x1000usize as *const ();
        let w1 = 0x2000usize as PyObjectRef;
        let w2 = 0x3000usize as PyObjectRef;
        register_live_code_wrapper(code, w1);
        // A later authoritative GC edge refreshes the movable wrapper address.
        register_live_code_wrapper(code, w2);
        assert_eq!(live_code_wrapper(code), w2);
        // An unregistered code pointer recovers to PY_NULL.
        assert!(live_code_wrapper(0x9999usize as *const ()).is_null());
        // Null inputs are no-ops / recover to PY_NULL.
        register_live_code_wrapper(std::ptr::null(), w1);
        assert!(live_code_wrapper(std::ptr::null()).is_null());
    }

    #[test]
    fn w_code_const_null_code_ptr_returns_py_null() {
        // A `PyCode` built from a null `code_ptr` must not be
        // dereferenced; the guard returns PY_NULL so the caller falls back to
        // its own constant realization. Construction still publishes an empty
        // `co_consts_w` so `getconstant_w` is an array read.
        let w_code = w_code_new(std::ptr::null());
        let table = unsafe { (*(w_code as *const PyCode)).co_consts_w };
        assert!(!table.is_null());
        assert_eq!(unsafe { (*table).len() }, 0);
        let result = unsafe { w_code_const(w_code, 0) };
        assert_eq!(result, pyre_object::pyobject::PY_NULL);
    }

    /// `PyCode_Addr2Line` on a real code object: a negative offset names the
    /// first line, an in-range offset names the line its range carries, and an
    /// offset past the last range reports no line at all.
    #[test]
    fn w_code_addr2line_halves_the_offset_and_reports_a_miss() {
        let code = compile_exec("a = 1\nb = 2\nc = 3\n").expect("compile failed");
        let firstlineno = code.first_line_number.map_or(1, |line| line.get()) as i32;
        assert_eq!(code_addr2line(&code, firstlineno, -1), firstlineno);
        assert_eq!(code_addr2line(&code, firstlineno, -2), firstlineno);

        let count = code.instructions.len();
        assert!(count > 0);
        let lines: Vec<i32> = (0..count)
            .map(|index| code_addr2line(&code, firstlineno, index as i64 * 2))
            .collect();
        for (index, line) in lines.iter().copied().enumerate() {
            // The odd byte inside an instruction reads that instruction's row.
            assert_eq!(
                code_addr2line(&code, firstlineno, index as i64 * 2 + 1),
                line
            );
        }
        // Each of the three statements contributes at least one instruction.
        for statement in 0..3 {
            assert!(lines.contains(&(firstlineno + statement)), "{lines:?}");
        }

        assert_eq!(code_addr2line(&code, firstlineno, 1 << 30), -1);
        // An empty line table describes no range, so nothing resolves against
        // it — `linetable_to_locations` instead expands one first-line row per
        // instruction, which is why this reads `linetable` directly.
        let mut blank = compile_exec("a = 1\n").expect("compile failed");
        blank.linetable = Vec::new().into_boxed_slice();
        assert_eq!(code_addr2line(&blank, 1, 0), -1);
        assert_eq!(code_addr2line(&blank, 1, 2), -1);
        // A negative `co_firstlineno` is answered as-is, not clamped: only
        // `code.replace` rejects one, `CodeType(...)` stores it.
        assert_eq!(code_addr2line(&blank, -8, -1), -8);
        assert_eq!(code_addr2line(&blank, 0, -1), 0);
    }

    /// Marshal and `code.replace` leave `locations` empty. The rows come back
    /// from `linetable` on the first reader, matching the table the compiler
    /// expanded up front.
    #[test]
    fn co_names_w_and_deferred_locations() {
        {
            // deferred_locations_rebuild_from_the_line_table
            let mut code = compile_exec("a = 1\nb = 2\n").expect("compile failed");
            let expanded = code.locations.to_vec();
            assert!(!expanded.is_empty());
            code.locations = Vec::new().into_boxed_slice();
            let firstlineno = code.first_line_number.map_or(1, |line| line.get()) as i32;
            release_code_locations(&mut code, firstlineno);
            assert!(code.locations.is_empty());
            assert_eq!(code_locations(&code), expanded.as_slice());
            code_locations_cache()
                .lock()
                .remove(&(&code as *const crate::CodeObject as usize));
        }
        {
            // co_names_w_matches_pycode_init
            let code = compile_exec("answer = 42\n").expect("compile failed");
            let idx = code
                .names
                .iter()
                .position(|name| name == "answer")
                .expect("answer");
            let w_code = box_code_object(code);
            let py = unsafe { &*(w_code as *const PyCode) };
            let slot = unsafe { &*py.co_names_w };
            let published = slot[idx];
            assert!(!published.is_null());
            assert_eq!(
                published,
                pyre_object::unicodeobject::intern_str_value("answer")
            );
        }
        {
            // interned_co_names_keep_the_marshal_object
            let module = compile_exec("def f(x):\n    return len(x)\n").expect("compile");
            let code = module
                .constants
                .iter()
                .find_map(|constant| match constant {
                    crate::bytecode::ConstantData::Code { code } => Some((**code).clone()),
                    _ => None,
                })
                .expect("function");
            assert!(!code.names.is_empty(), "co_names");
            let _roots = pyre_object::gc_roots::push_roots();
            let mut slots = Vec::new();
            for name in &code.names {
                let obj = unsafe { pyre_object::unicodeobject::intern_str_value(name.as_ref()) };
                let slot = pyre_object::gc_roots::shadow_stack_len();
                let _ = pyre_object::gc_roots::pin_root(obj);
                slots.push(slot);
            }
            let expected = pyre_object::gc_roots::shadow_stack_get(slots[0]);
            let w_code = box_code_object_with_interned_name_slots(code, &slots);
            let stored = unsafe {
                let py = &*(w_code as *const PyCode);
                assert!(!py.co_names_w.is_null());
                (*py.co_names_w).as_slice()[0]
            };
            assert!(std::ptr::eq(stored, expected));
        }
    }

    /// The entry payload is peeked, not consumed: `advance` moves by the
    /// bit-7 scan alone, so a byte that is a payload by length and a header by
    /// its marker starts the next range.
    #[test]
    fn advance_stops_at_the_marker_a_short_payload_would_swallow() {
        let table = [0x00u8, 0x80u8];
        let mut bounds = PyCodeAddressRange::new(&table, 3);
        assert!(bounds.advance());
        assert_eq!((bounds.ar_start, bounds.ar_end, bounds.ar_line), (0, 2, 3));
        assert!(bounds.advance());
        assert_eq!((bounds.ar_start, bounds.ar_end, bounds.ar_line), (2, 4, 3));
        assert!(!bounds.advance());
    }

    #[test]
    fn pycode_co_flags_matches_generator_code_bits() {
        let code = compile_exec("def f():\n    yield 1\n").expect("compile failed");
        let nested = code
            .constants
            .iter()
            .find_map(|constant| match constant {
                crate::bytecode::ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("generator code");
        let bits = nested.flags.bits();
        let w_code = box_code_object(nested);
        let pycode = unsafe { &*(w_code as *const PyCode) };
        assert_ne!(
            pycode.co_flags & crate::astcompiler::consts::CO_GENERATOR,
            0
        );
        assert_eq!(pycode.co_flags, bits);
    }

    #[test]
    fn box_code_object_preserves_owned_storage() {
        let code = compile_exec("answer = 42\n").expect("compile failed");
        let source_storage = code.source_path.as_ptr();
        let instruction_storage = code.instructions.as_ptr();

        let w_code = box_code_object(code);
        let stored = unsafe { &*(w_code_get_ptr(w_code) as *const crate::CodeObject) };

        assert_eq!(stored.source_path.as_ptr(), source_storage);
        assert_eq!(stored.instructions.as_ptr(), instruction_storage);
        let w_filename = unsafe { (*(w_code as *const PyCode)).w_filename };
        assert!(!w_filename.is_null());
        assert!(std::ptr::eq(
            unsafe { w_code_filename_obj(w_code) },
            w_filename
        ));
        assert_eq!(
            unsafe { pyre_object::unicodeobject::w_str_get_value_opt(w_filename) },
            Some(stored.source_path.as_str())
        );
    }

    #[test]
    fn a_code_graph_is_retired_by_its_last_wrapper_and_never_before() {
        // Two nested functions, so the graph has members the top-level
        // wrapper does not speak for.
        let code = compile_exec(
            "def outer():\n    def inner():\n        return 'body'\n    return inner\n",
        )
        .expect("compile failed");
        let w_code = box_code_object(code);
        let owner = unsafe { w_code_get_ptr(w_code) } as usize;

        let nested: Vec<usize> = {
            let registry = code_units()
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let unit = registry.units.get(&owner).expect("owner registered");
            // One wrapper per member: the top-level plus every nested code
            // constant `w_code_new_owned`'s fill loop realized.
            assert_eq!(unit.wrappers, unit.members.len());
            assert!(unit.members.len() >= 3, "{:?}", unit.members);
            assert!(!unit.pinned);
            unit.members
                .iter()
                .copied()
                .filter(|member| *member != owner)
                .collect()
        };

        let body = || unsafe { &*(owner as *const crate::CodeObject) };
        for member in &nested {
            detach_code_unit_wrapper(*member as *const ());
            assert!(
                !body().instructions.is_empty(),
                "the graph was retired while {member:#x}'s siblings were still live"
            );
        }
        detach_code_unit_wrapper(owner as *const ());
        assert!(body().instructions.is_empty());
        assert!(body().names.is_empty());
        assert!(body().source_path.is_empty());
        // The shells stay allocated: a raw body address is the key of every
        // side table in the tree, so none may be handed back to the allocator.
        for member in &nested {
            let nested_body = unsafe { &*(*member as *const crate::CodeObject) };
            assert!(nested_body.instructions.is_empty());
        }
        let registry = code_units()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(!registry.units.contains_key(&owner));
        assert!(registry.owner_of.keys().all(|key| !nested.contains(key)));
    }

    #[test]
    fn a_pinned_code_graph_survives_its_last_wrapper() {
        let code = compile_exec("def f():\n    return 'kept'\n").expect("compile failed");
        let w_code = box_code_object(code);
        let owner = unsafe { w_code_get_ptr(w_code) } as usize;
        pin_code_unit(owner as *const ());

        let members: Vec<usize> = code_units()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .units
            .get(&owner)
            .expect("owner registered")
            .members
            .clone();
        for member in &members {
            detach_code_unit_wrapper(*member as *const ());
        }
        let body = unsafe { &*(owner as *const crate::CodeObject) };
        assert!(
            !body.instructions.is_empty(),
            "a jitcode's raw body address must stay readable"
        );
    }

    #[test]
    fn an_unowned_wrapper_takes_no_part_in_retirement() {
        // `w_code_new`'s contract is that the body is never released, so its
        // wrappers register nothing and detaching one retires nothing.
        let code = compile_exec("x = 1\n").expect("compile failed");
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        let _w_code = w_code_new(code_ptr);
        assert_eq!(code_unit_owner(code_ptr), 0);
        detach_code_unit_wrapper(code_ptr);
        let body = unsafe { &*(code_ptr as *const crate::CodeObject) };
        assert!(!body.instructions.is_empty());
    }

    #[test]
    fn w_code_const_shares_large_integer_and_root_walker_visits_slot() {
        let code =
            compile_exec("x = 123456789012345678901234567890123456789012345678901234567890\n")
                .expect("compile failed");
        let idx = code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("large integer constant");
        let w_code = box_code_constant(&code);

        let eager = unsafe { (&*(*(w_code as *const PyCode)).co_consts_w)[idx] };
        assert_ne!(
            eager,
            pyre_object::pyobject::PY_NULL,
            "PyCode construction must eagerly wrap every compiler constant"
        );
        let first = unsafe { w_code_const(w_code, idx) };
        let second = unsafe { w_code_const(w_code, idx) };
        assert_eq!(first, eager);
        assert_eq!(
            first, second,
            "getconstant_w must return the co_consts_w slot identity"
        );
        assert!(unsafe { pyre_object::is_long(first) });
        let exposed = unsafe { code_get_field(w_code, "co_consts") }.expect("co_consts descriptor");
        assert_eq!(
            unsafe { pyre_object::w_tuple_getitem(exposed, idx as i64) },
            Some(first),
            "code.co_consts must expose the same wrapped slot object"
        );

        // The items are the array's own edges; the code reports the array.
        let table = unsafe { (*(w_code as *const PyCode)).co_consts_w } as usize;
        let mut visited = false;
        unsafe {
            crate::eval::walk_raw_code_roots(w_code, &mut |root| {
                if root.0 == table {
                    visited = true;
                }
            });
        }
        assert!(
            visited,
            "managed PyCode must expose its co_consts_w array as a root"
        );
    }

    #[test]
    fn copy_const_slots_preserves_eager_source_identities() {
        let code = compile_exec(
            "x = 12345678901234567890123456789012345678901234567890\n\
             y = 98765432109876543210987654321098765432109876543210\n",
        )
        .expect("compile failed");
        let integer_indices: Vec<usize> = code
            .constants
            .iter()
            .enumerate()
            .filter_map(|(index, constant)| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
                .then_some(index)
            })
            .collect();
        assert!(integer_indices.len() >= 2);
        let first_idx = integer_indices[0];
        let second_idx = integer_indices[1];
        let src = box_code_constant(&code);
        let dst = box_code_constant(&code);
        let first = unsafe { w_code_const(src, first_idx) };
        let second = unsafe { w_code_const(src, second_idx) };

        unsafe {
            w_code_copy_const_slots(dst, src);
            let dst_slots = &*(*(dst as *const PyCode)).co_consts_w;
            assert_eq!(dst_slots[first_idx], first);
            assert_eq!(
                dst_slots[second_idx], second,
                "code.replace must preserve every eager co_consts_w identity"
            );
        }
    }

    #[test]
    fn raw_code_root_walker_reports_one_gc_edge_at_a_time() {
        let code = compile_exec(
            "def outer():\n\
             \x20   def inner():\n\
             \x20       return 12345678901234567890123456789012345678901234567890\n\
             \x20   return inner\n",
        )
        .expect("compile failed");
        let (outer_idx, outer_code) = code
            .constants
            .iter()
            .enumerate()
            .find_map(|(index, constant)| match constant {
                crate::bytecode::ConstantData::Code { code } => Some((index, code.as_ref())),
                _ => None,
            })
            .expect("outer code constant");
        let (inner_idx, inner_code) = outer_code
            .constants
            .iter()
            .enumerate()
            .find_map(|(index, constant)| match constant {
                crate::bytecode::ConstantData::Code { code } => Some((index, code.as_ref())),
                _ => None,
            })
            .expect("inner code constant");
        let bigint_idx = inner_code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("nested large integer");

        let w_top = box_code_constant(&code);
        let w_outer = unsafe { w_code_const(w_top, outer_idx) };
        let w_inner = unsafe { w_code_const(w_outer, inner_idx) };
        let w_bigint = unsafe { w_code_const(w_inner, bigint_idx) };
        let top_table = unsafe { (*(w_top as *const PyCode)).co_consts_w } as usize;
        let inner_table = unsafe { (*(w_inner as *const PyCode)).co_consts_w } as usize;
        let mut top_reaches_table = false;
        let mut top_reaches_item = false;
        let mut top_reaches_deep_value = false;
        unsafe {
            crate::eval::walk_raw_code_roots(w_top, &mut |root| {
                if root.0 == top_table {
                    top_reaches_table = true;
                }
                if root.0 == w_outer as usize {
                    top_reaches_item = true;
                }
                if root.0 == w_bigint as usize {
                    top_reaches_deep_value = true;
                }
            });
        }
        assert!(
            top_reaches_table,
            "a PyCode trace must report its co_consts_w array"
        );
        assert!(
            !top_reaches_item && !top_reaches_deep_value,
            "a GC trace callback must leave the array's items and transitive \
             traversal to the mark worklist"
        );
        let mut inner_reaches_table = false;
        unsafe {
            crate::eval::walk_raw_code_roots(w_inner, &mut |root| {
                if root.0 == inner_table {
                    inner_reaches_table = true;
                }
            });
        }
        assert!(
            inner_reaches_table,
            "the nested object's direct edge was lost"
        );
    }

    #[test]
    fn w_code_const_reads_are_free_threaded_identity_safe() {
        let code =
            compile_exec("x = 314159265358979323846264338327950288419716939937510582097494\n")
                .expect("compile failed");
        let idx = code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("large integer constant");
        let w_code = box_code_constant(&code) as usize;
        let canonical = unsafe { w_code_const(w_code as PyObjectRef, idx) as usize };
        assert_ne!(canonical, 0, "constructor publishes co_consts_w[index]");
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                unsafe { w_code_const(w_code as PyObjectRef, idx) as usize }
            }));
        }
        let values: Vec<usize> = workers
            .into_iter()
            .map(|worker| worker.join().expect("constant worker panicked"))
            .collect();
        assert!(
            values.iter().all(|value| *value == canonical),
            "all readers must observe one canonical co_consts_w wrapper"
        );
    }

    fn co_names_table(w_code: PyObjectRef) -> *mut FixedObjectArray {
        let table = unsafe { (*(w_code as *const PyCode)).co_names_w };
        assert!(
            !table.is_null(),
            "co_names_w is never null (pycode.py PyCode.__init__ empty list)"
        );
        table
    }

    fn nested_code(source: &str) -> crate::CodeObject {
        compile_exec(source)
            .expect("compile failed")
            .constants
            .iter()
            .find_map(|constant| match constant {
                crate::bytecode::ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("nested code")
    }

    #[test]
    fn co_names_w_slots_are_interned_at_construction() {
        let source = "def f(a):\n    return a.x + global_name\n";
        let w_code = box_code_object(compile_exec(source).expect("compile failed"));
        let w_code_again = box_code_object(compile_exec(source).expect("compile failed"));
        let table = co_names_table(w_code);
        let slots = unsafe { &*table };
        let again = unsafe { &*co_names_table(w_code_again) };
        assert!(!slots.is_empty());
        assert_eq!(slots.len(), again.len());
        for (index, (left, right)) in slots.as_slice().iter().zip(again.as_slice()).enumerate() {
            assert!(!left.is_null());
            assert_eq!(left, right);
            assert!(unsafe { pyre_object::unicodeobject::is_interned_exact_str(*left) });
            assert_eq!(unsafe { w_code_getname_w(w_code, index) }, *left);
        }
        let replaced = unsafe { code_replace(&[w_code]).expect("replace") };
        let copied_table = co_names_table(replaced);
        let copied = unsafe { &*copied_table };
        for (src, dst) in slots.as_slice().iter().zip(copied.as_slice()) {
            assert_eq!(src, dst);
        }
        let stub = w_code_new(std::ptr::null());
        assert_eq!(unsafe { (*co_names_table(stub)).len() }, 0);
        assert!(unsafe { !w_code_name_index_covered(stub, 0) });
        let minted = unsafe { w_code_getname_w_or_new(stub, 0, "x") };
        assert_eq!(minted, pyre_object::unicodeobject::intern_str_value("x"));
        let unaligned = w_code_new(0xDEAD_BEEF as *const ());
        assert_eq!(unsafe { (*co_names_table(unaligned)).len() }, 0);
        assert!(unsafe { !w_code_name_index_covered(unaligned, 0) });
    }

    #[test]
    fn co_names_w_empty_replace_and_fill_keep_the_gc_table() {
        // `(lambda: 1).__code__` / `def f(): return 1` has empty `co_names`.
        let empty = box_code_object(nested_code("def f():\n    return 1\n"));
        let empty_table = co_names_table(empty);
        assert_eq!(unsafe { (*empty_table).len() }, 0);
        let replaced = unsafe { code_replace(&[empty]).expect("replace empty names") };
        let replaced_table = co_names_table(replaced);
        assert_eq!(unsafe { (*replaced_table).len() }, 0);
        let empty_tuple = pyre_object::w_tuple_new(Vec::new());
        unsafe { w_code_fill_names_from_tuple(empty, empty_tuple) };
        assert_eq!(co_names_table(empty), empty_table);
        unsafe { w_code_copy_name_slots(empty, replaced) };
        assert_eq!(co_names_table(empty), empty_table);

        let named = box_code_object(nested_code("def g():\n    return a + b\n"));
        let named_table = co_names_table(named);
        assert_eq!(unsafe { (*named_table).len() }, 2);
        let first = pyre_object::unicodeobject::intern_str_value("x");
        let second = pyre_object::unicodeobject::intern_str_value("y");
        let names = pyre_object::w_tuple_new(vec![first, second]);
        unsafe { w_code_fill_names_from_tuple(named, names) };
        assert_eq!(
            co_names_table(named),
            named_table,
            "fill writes the existing GC table, it does not replace the allocation"
        );
        let slots = unsafe { &*named_table };
        assert_eq!(slots[0], first);
        assert_eq!(slots[1], second);
        let copy_dst = box_code_object(nested_code("def h():\n    return a + b\n"));
        let copy_table = co_names_table(copy_dst);
        unsafe { w_code_copy_name_slots(copy_dst, named) };
        assert_eq!(co_names_table(copy_dst), copy_table);
        let copied = unsafe { &*copy_table };
        assert_eq!(copied[0], first);
        assert_eq!(copied[1], second);
    }
}
