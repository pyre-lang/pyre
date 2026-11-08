//! `rpython/rtyper/tool/rffi_platform.py`: ask the C compiler from a
//! build script. Compilation uses `crate::gcc_cache`
//! (`build_executable_cache`, `try_compile_cache`) and
//! `crate::platform::CompilationError`; probe sources go under
//! `crate::udir::udir()` (`uniquefilepath`).

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

use crate::ExternalCompilationInfo;
pub use crate::platform::CompilationError;

/// `rffi_platform.py` `C_HEADER`.
const C_HEADER: &str = r#"
#include <stdio.h>
#include <stddef.h>   /* for offsetof() */

void dump(char* key, int value) {
    printf("%s: %d\n", key, value);
}
"#;

/// The (sign, bits) class, `rffi.size_and_sign`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntType {
    pub signed: bool,
    pub bits: u32,
}

/// A field type for `Struct` interesting_fields and the resulting layout.
/// `rffi_platform.py` `integer_class` / `float_class` / `lltype.FixedSizeArray`
/// / `rffi.CArray` / nested `Struct`.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldType {
    Int(IntType),
    Double,
    /// `lltype.FixedSizeArray(OF, length)`.
    FixedSizeArray {
        of: Box<FieldType>,
        length: usize,
    },
    /// `rffi.CArray` / `lltype.Array` with `hints={'nolength': True}`.
    Array {
        of: Box<FieldType>,
    },
    /// Nested `Struct` keyed by C name among this `CConfig`'s entries
    /// (`Struct.build_result` `isinstance(fieldtype, Struct)`).
    Nested {
        c_name: String,
    },
    /// A built `Struct.build_result` / `rffi.CStruct`.
    Struct(Rc<StructLayout>),
}

impl FieldType {
    /// `rffi.INT`.
    pub fn int() -> Self {
        FieldType::Int(IntType {
            signed: true,
            bits: 32,
        })
    }

    /// `rffi.UINT`.
    pub fn uint() -> Self {
        FieldType::Int(IntType {
            signed: false,
            bits: 32,
        })
    }

    /// `rffi.USHORT`.
    pub fn ushort() -> Self {
        FieldType::Int(IntType {
            signed: false,
            bits: 16,
        })
    }

    /// `lltype.Signed` (`rffi.LONG` on LP64).
    pub fn signed_word() -> Self {
        FieldType::Int(IntType {
            signed: true,
            bits: (std::mem::size_of::<usize>() * 8) as u32,
        })
    }

    /// `rffi.UCHAR`.
    pub fn uchar() -> Self {
        FieldType::Int(IntType {
            signed: false,
            bits: 8,
        })
    }

    /// `rffi.CHAR` / `lltype.Char`.
    pub fn char() -> Self {
        FieldType::Int(IntType {
            signed: false,
            bits: 8,
        })
    }

    /// `rffi.DOUBLE`.
    pub fn double() -> Self {
        FieldType::Double
    }

    /// Nested `Struct` field (`interesting_fields` stores the `Struct` entry).
    pub fn nested(c_name: impl Into<String>) -> Self {
        FieldType::Nested {
            c_name: c_name.into(),
        }
    }

    /// `lltype.FixedSizeArray(of, length)`.
    pub fn fixed_array(of: FieldType, length: usize) -> Self {
        FieldType::FixedSizeArray {
            of: Box::new(of),
            length,
        }
    }

    /// `rffi.CArray(of)`.
    pub fn array(of: FieldType) -> Self {
        FieldType::Array { of: Box::new(of) }
    }

    /// `fieldtype in integer_class`.
    fn is_integer(&self) -> bool {
        matches!(self, FieldType::Int(_))
    }
}

/// `rffi.CStruct` produced by `Struct.build_result`.
#[derive(Clone, Debug)]
pub struct StructLayout {
    /// `CStruct` name: `struct ` stripped.
    pub name: String,
    /// Original `Struct.name` (`PaddingDrop`).
    pub c_name: String,
    /// `hints['typedef']` when the C name has no `struct ` prefix.
    pub typedef: bool,
    pub align: u64,
    pub size: u64,
    /// `c_`-prefixed names (`CStruct`).
    pub fields: Vec<(String, FieldType)>,
    pub fieldoffsets: Vec<u64>,
    pub padding: Vec<String>,
    pub eci: ExternalCompilationInfo,
    padding_drop_cache: RefCell<Option<Vec<String>>>,
}

impl PartialEq for StructLayout {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.c_name == other.c_name
            && self.typedef == other.typedef
            && self.align == other.align
            && self.size == other.size
            && self.fields == other.fields
            && self.fieldoffsets == other.fieldoffsets
            && self.padding == other.padding
    }
}

impl StructLayout {
    /// `lltype.Struct` field type (`S.c_d_reclen`).
    pub fn c_field(&self, name: &str) -> &FieldType {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
            .unwrap_or_else(|| panic!("AttributeError: {name}"))
    }

    /// `rffi.offsetof(STRUCT, fieldname)`.
    pub fn offsetof(&self, fieldname: &str) -> u64 {
        self.fields
            .iter()
            .zip(&self.fieldoffsets)
            .find(|((n, _), _)| n == fieldname)
            .map(|(_, o)| *o)
            .unwrap_or_else(|| panic!("offsetof: {fieldname}"))
    }

    /// `hints['get_padding_drop']` / `PaddingDrop.__call__`.
    pub fn get_padding_drop(
        &self,
        types: &[(String, String)],
    ) -> Result<Vec<String>, CompilationError> {
        if let Some(cache) = self.padding_drop_cache.borrow().as_ref() {
            return Ok(cache.clone());
        }
        let allfields: Vec<String> = self.fields.iter().map(|(n, _)| n.clone()).collect();
        let drops =
            padding_drop_compute_now(&self.c_name, &allfields, &self.padding, &self.eci, types)?;
        *self.padding_drop_cache.borrow_mut() = Some(drops.clone());
        Ok(drops)
    }

    /// Spell the `rffi.CStruct` that `Struct.build_result` returns as a
    /// `#[repr(C)]` Rust struct. `nested` maps a nested struct's
    /// `StructLayout.name` to its Rust name.
    pub fn rust_struct_source(&self, rust_name: &str, nested: &[(&str, &str)]) -> String {
        let mut out = String::new();
        out.push_str("#[repr(C)]\n");
        out.push_str("#[derive(Clone, Copy)]\n");
        out.push_str(&format!("pub struct {rust_name} {{\n"));
        for (fname, fty) in &self.fields {
            out.push_str(&format!(
                "    pub {fname}: {},\n",
                field_type_rust(fty, nested)
            ));
        }
        out.push_str("}\n");
        out.push_str(&format!(
            "const _: () = assert!(::core::mem::size_of::<{rust_name}>() == {} && ::core::mem::align_of::<{rust_name}>() == {});\n",
            self.size, self.align
        ));
        out
    }
}

/// Rust type spelled by `StructLayout::rust_struct_source` for one field.
fn field_type_rust(tp: &FieldType, nested: &[(&str, &str)]) -> String {
    match tp {
        FieldType::Int(IntType { signed, bits }) => {
            if *signed {
                format!("i{bits}")
            } else {
                format!("u{bits}")
            }
        }
        FieldType::Double => "f64".to_string(),
        FieldType::FixedSizeArray { of, length } => {
            format!("[{}; {length}]", field_type_rust(of, nested))
        }
        FieldType::Array { of } => format!("[{}; 0]", field_type_rust(of, nested)),
        FieldType::Struct(s) => nested
            .iter()
            .find(|(name, _)| *name == s.name)
            .map(|(_, rust)| (*rust).to_string())
            .unwrap_or_else(|| panic!("nested Struct {} has no Rust name", s.name)),
        FieldType::Nested { c_name } => {
            panic!("build_result has resolved Nested {c_name}")
        }
    }
}

/// `rffi_platform.py` `Field`.
struct LayoutField {
    name: String,
    ctype: FieldType,
}

/// `rffi_platform.py` `CConfigEntry`.
pub enum CConfigEntry {
    /// `rffi_platform.py` `SimpleType`. The `ctype_hint` is `rffi.INT`
    /// (4-byte signed); `fixup_ctype` lands on the measured (size, sign)
    /// when that differs. `ifdef` is `SimpleType(..., ifdef=)`.
    SimpleType { name: String, ifdef: Option<String> },
    /// `rffi_platform.py` `SizeOf`.
    SizeOf(String),
    /// `rffi_platform.py` `ConstantInteger`.
    ConstantInteger(String),
    /// `rffi_platform.py` `IntegerFunctionResult`.
    IntegerFunctionResult { name: String, args: Vec<String> },
    /// `rffi_platform.py` `DefinedConstantInteger`.
    DefinedConstantInteger(String),
    /// `rffi_platform.py` `DefinedConstantDouble`.
    DefinedConstantDouble(String),
    /// `rffi_platform.py` `DefinedConstantString`. `macro` is the `#ifdef`
    /// name; `name` is the C expression (`DefinedConstantString(macro, name)`).
    DefinedConstantString { macro_name: String, name: String },
    /// `rffi_platform.py` `Defined`.
    Defined(String),
    /// `rffi_platform.py` `Struct`.
    Struct {
        name: String,
        interesting_fields: Vec<(String, FieldType)>,
        ifdef: Option<String>,
    },
    /// `rffi_platform.py` `_PaddingDropFieldLookup`.
    PaddingDropFieldLookup {
        name: String,
        staticfields: Vec<Option<String>>,
        fieldname: String,
    },
}

impl CConfigEntry {
    /// `SimpleType(name)` with `ctype_hint=rffi.INT` and `ifdef=None`.
    pub fn simple_type(name: impl Into<String>) -> Self {
        CConfigEntry::SimpleType {
            name: name.into(),
            ifdef: None,
        }
    }

    /// `Struct(name, interesting_fields, ifdef=None)`.
    pub fn struct_type(
        name: impl Into<String>,
        interesting_fields: Vec<(String, FieldType)>,
        ifdef: Option<String>,
    ) -> Self {
        CConfigEntry::Struct {
            name: name.into(),
            interesting_fields,
            ifdef,
        }
    }

    fn prepare_code(&self) -> Vec<String> {
        match self {
            CConfigEntry::SimpleType { name, ifdef } => {
                let mut lines = Vec::new();
                if let Some(ifdef) = ifdef {
                    lines.push(format!("#ifdef {ifdef}"));
                }
                lines.push(format!("typedef {name} platcheck_t;"));
                lines.push(String::new());
                lines.push("platcheck_t x;".to_string());
                if ifdef.is_some() {
                    lines.push("dump(\"defined\", 1);".to_string());
                }
                lines.push("dump(\"size\",  sizeof(platcheck_t));".to_string());
                lines.push("x = 0; x = ~x;".to_string());
                lines.push("dump(\"unsigned\", x > 0);".to_string());
                if ifdef.is_some() {
                    lines.push("#else".to_string());
                    lines.push("dump(\"defined\", 0);".to_string());
                    lines.push("#endif".to_string());
                }
                lines
            }
            CConfigEntry::SizeOf(name) => {
                vec![format!("dump(\"size\",  sizeof({name}));")]
            }
            CConfigEntry::ConstantInteger(name) => integer_value_code(name),
            CConfigEntry::IntegerFunctionResult { name, args } => {
                let args = args.join(", ");
                let mut lines = vec![format!("long int result = {name}({args});")];
                lines.extend(integer_value_code("result"));
                lines
            }
            CConfigEntry::DefinedConstantInteger(macro_name) => {
                let mut lines = vec![format!("#ifdef {macro_name}")];
                lines.push("dump(\"defined\", 1);".to_string());
                lines.extend(integer_value_code(macro_name));
                lines.push("#else".to_string());
                lines.push("dump(\"defined\", 0);".to_string());
                lines.push("#endif".to_string());
                lines
            }
            CConfigEntry::DefinedConstantDouble(macro_name) => vec![
                format!("#ifdef {macro_name}"),
                "int i;".to_string(),
                format!("double x = {macro_name};"),
                "unsigned char *p = (unsigned char *)&x;".to_string(),
                "dump(\"defined\", 1);".to_string(),
                "for (i = 0; i < 8; i++) {".to_string(),
                " printf(\"value_%d: %d\\n\", i, p[i]);".to_string(),
                "}".to_string(),
                "#else".to_string(),
                "dump(\"defined\", 0);".to_string(),
                "#endif".to_string(),
            ],
            CConfigEntry::DefinedConstantString { macro_name, name } => vec![
                format!("#ifdef {macro_name}"),
                "int i;".to_string(),
                format!("const char *p = {name};"),
                "dump(\"defined\", 1);".to_string(),
                "for (i = 0; p[i] != 0; i++ ) {".to_string(),
                "  printf(\"value_%d: %d\\n\", i, (int)(unsigned char)p[i]);".to_string(),
                "}".to_string(),
                "#else".to_string(),
                "dump(\"defined\", 0);".to_string(),
                "#endif".to_string(),
            ],
            CConfigEntry::Defined(macro_name) => vec![
                format!("#ifdef {macro_name}"),
                "dump(\"defined\", 1);".to_string(),
                "#else".to_string(),
                "dump(\"defined\", 0);".to_string(),
                "#endif".to_string(),
            ],
            CConfigEntry::Struct {
                name,
                interesting_fields,
                ifdef,
            } => {
                let mut lines = Vec::new();
                if let Some(ifdef) = ifdef {
                    lines.push(format!("#ifdef {ifdef}"));
                }
                lines.push(format!("typedef {name} platcheck_t;"));
                lines.push("typedef struct {".to_string());
                lines.push("    char c;".to_string());
                lines.push("    platcheck_t s;".to_string());
                lines.push("} platcheck2_t;".to_string());
                lines.push(String::new());
                lines.push("platcheck_t s;".to_string());
                if ifdef.is_some() {
                    lines.push("dump(\"defined\", 1);".to_string());
                }
                lines.push("dump(\"align\", offsetof(platcheck2_t, s));".to_string());
                lines.push("dump(\"size\",  sizeof(platcheck_t));".to_string());
                for (fieldname, fieldtype) in interesting_fields {
                    lines.push(format!(
                        "dump(\"fldofs {fieldname}\", offsetof(platcheck_t, {fieldname}));"
                    ));
                    lines.push(format!(
                        "dump(\"fldsize {fieldname}\",   sizeof(s.{fieldname}));"
                    ));
                    if fieldtype.is_integer() {
                        lines.push(format!(
                            "s.{fieldname} = 0; s.{fieldname} = ~s.{fieldname};"
                        ));
                        lines.push(format!(
                            "dump(\"fldunsigned {fieldname}\", s.{fieldname} > 0);"
                        ));
                    }
                }
                if ifdef.is_some() {
                    lines.push("#else".to_string());
                    lines.push("dump(\"defined\", 0);".to_string());
                    lines.push("#endif".to_string());
                }
                lines
            }
            CConfigEntry::PaddingDropFieldLookup {
                name,
                staticfields,
                fieldname,
            } => {
                let mut lines = vec![
                    format!("typedef {name} platcheck_t;"),
                    "static platcheck_t s = {".to_string(),
                ];
                for (i, ty) in staticfields.iter().enumerate() {
                    let value = if i == staticfields.len() - 1 { -1 } else { 0 };
                    match ty {
                        Some(ty) => lines.push(format!("\t({ty}){value},")),
                        None => lines.push(format!("\t{value},")),
                    }
                }
                lines.push("};".to_string());
                assert!(fieldname.starts_with("c_"), "{fieldname}");
                let c_field = &fieldname[2..];
                lines.push(format!("dump(\"fieldlookup\", s.{c_field} != 0);"));
                lines
            }
        }
    }

    fn build_result(
        &self,
        info: &[(String, i128)],
        config_result: Option<&ConfigResult<'_>>,
    ) -> Result<ConfigValue, CompilationError> {
        let get = |key: &str| info_get(info, key);
        match self {
            CConfigEntry::SimpleType { name, ifdef } => {
                if ifdef.is_some() && get("defined") == 0 {
                    return Ok(ConfigValue::None);
                }
                let size = get("size");
                let unsigned = get("unsigned") != 0;
                // `ctype_hint` is `rffi.INT` (4-byte signed). Keep it when
                // `(size, sign)` matches; otherwise `fixup_ctype`.
                let ctype = if size == 4 && !unsigned {
                    IntType {
                        signed: true,
                        bits: 32,
                    }
                } else {
                    fixup_ctype(name, size, unsigned)?
                };
                Ok(ConfigValue::Type(ctype))
            }
            CConfigEntry::SizeOf(_) => Ok(ConfigValue::Size(get("size") as u64)),
            CConfigEntry::ConstantInteger(_) | CConfigEntry::IntegerFunctionResult { .. } => {
                Ok(ConfigValue::Int(expose_value_as_rpython(get("value"))?))
            }
            CConfigEntry::DefinedConstantInteger(_) => {
                if get("defined") != 0 {
                    Ok(ConfigValue::Int(expose_value_as_rpython(get("value"))?))
                } else {
                    Ok(ConfigValue::None)
                }
            }
            CConfigEntry::DefinedConstantDouble(_) => {
                if get("defined") != 0 {
                    let mut bytes = [0u8; 8];
                    for (i, slot) in bytes.iter_mut().enumerate() {
                        *slot = get(&format!("value_{i}")) as u8;
                    }
                    Ok(ConfigValue::Double(f64::from_ne_bytes(bytes)))
                } else {
                    Ok(ConfigValue::None)
                }
            }
            CConfigEntry::DefinedConstantString { .. } => {
                if get("defined") != 0 {
                    let mut bytes = Vec::new();
                    let mut d = 0;
                    while let Some((_, value)) =
                        info.iter().find(|(k, _)| k == &format!("value_{d}"))
                    {
                        bytes.push(*value as u8);
                        d += 1;
                    }
                    Ok(ConfigValue::Str(
                        String::from_utf8(bytes).unwrap_or_else(|err| panic!("{err}")),
                    ))
                } else {
                    Ok(ConfigValue::None)
                }
            }
            CConfigEntry::Defined(_) => Ok(ConfigValue::Bool(get("defined") != 0)),
            CConfigEntry::Struct {
                name,
                interesting_fields,
                ifdef,
            } => struct_build_result(
                name,
                interesting_fields,
                ifdef.as_deref(),
                info,
                config_result.expect("Struct.build_result config_result"),
            ),
            CConfigEntry::PaddingDropFieldLookup { .. } => Ok(ConfigValue::Int(get("fieldlookup"))),
        }
    }
}

fn info_get(info: &[(String, i128)], key: &str) -> i128 {
    info.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| *v)
        .unwrap_or_else(|| panic!("KeyError: {key}"))
}

fn info_get_opt(info: &[(String, i128)], key: &str) -> Option<i128> {
    info.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
}

/// `ConstantInteger.prepare_code` / `IntegerFunctionResult` value dump.
fn integer_value_code(expr: &str) -> Vec<String> {
    vec![
        format!("if (({expr}) <= 0) {{"),
        format!("    long long x = (long long)({expr});"),
        "    printf(\"value: %lld\\n\", x);".to_string(),
        "} else {".to_string(),
        format!("    unsigned long long x = (unsigned long long)({expr});"),
        "    printf(\"value: %llu\\n\", x);".to_string(),
        "}".to_string(),
    ]
}

/// `rffi_platform.py` `integer_class` / `fixup_ctype`. Members are
/// `SIGNEDCHAR`, `UCHAR`, `CHAR`, `SHORT`, `USHORT`, `INT`, `UINT`,
/// `INT_real`, `UINT_real`, `LONG`, `ULONG`, `LONGLONG`, `ULONGLONG`
/// (1, 2, 4, 8 bytes; no 128-bit type).
fn fixup_ctype(name: &str, size: i128, unsigned: bool) -> Result<IntType, CompilationError> {
    let bits = match size {
        1 => 8,
        2 => 16,
        4 => 32,
        8 => 64,
        _ => {
            return Err(CompilationError {
                out: String::new(),
                err: format!("conflicting field type for {name}"),
            });
        }
    };
    Ok(IntType {
        signed: !unsigned,
        bits,
    })
}

/// `rffi_platform.py` `expose_value_as_rpython`.
fn expose_value_as_rpython(value: i128) -> Result<i128, CompilationError> {
    if value >= i64::MIN as i128 && value <= u64::MAX as i128 {
        Ok(value)
    } else {
        Err(CompilationError {
            out: String::new(),
            err: format!("value {value} does not fit into any RPython integer type"),
        })
    }
}

/// C `long` width: LLP64 32 on Windows, LP64 pointer width elsewhere.
fn c_long_bits() -> u32 {
    if cfg!(windows) {
        32
    } else {
        (std::mem::size_of::<usize>() * 8) as u32
    }
}

/// `rffi_platform.py` `integer_class`: SIGNEDCHAR, UCHAR, CHAR, SHORT, USHORT,
/// INT, UINT, INT_real, UINT_real, LONG, ULONG, LONGLONG, ULONGLONG.
fn integer_class() -> [IntType; 13] {
    let long_bits = c_long_bits();
    [
        IntType {
            signed: true,
            bits: 8,
        },
        IntType {
            signed: false,
            bits: 8,
        },
        IntType {
            signed: false,
            bits: 8,
        },
        IntType {
            signed: true,
            bits: 16,
        },
        IntType {
            signed: false,
            bits: 16,
        },
        IntType {
            signed: true,
            bits: 32,
        },
        IntType {
            signed: false,
            bits: 32,
        },
        IntType {
            signed: true,
            bits: 32,
        },
        IntType {
            signed: false,
            bits: 32,
        },
        IntType {
            signed: true,
            bits: long_bits,
        },
        IntType {
            signed: false,
            bits: long_bits,
        },
        IntType {
            signed: true,
            bits: 64,
        },
        IntType {
            signed: false,
            bits: 64,
        },
    ]
}

/// `rffi_platform.py` `_sizeof`.
fn sizeof_type(tp: &FieldType) -> usize {
    match tp {
        FieldType::Int(t) => (t.bits / 8) as usize,
        FieldType::Double => 8,
        FieldType::FixedSizeArray { of, length } => sizeof_type(of) * *length,
        FieldType::Array { .. } => panic!("_sizeof CArray"),
        FieldType::Nested { c_name } => panic!("_sizeof unresolved Nested {c_name}"),
        FieldType::Struct(s) => s.fields.iter().map(|(_, t)| sizeof_type(t)).sum(),
    }
}

/// `rffi.size_and_sign`.
fn size_and_sign(tp: &FieldType) -> (i128, bool) {
    let size = sizeof_type(tp) as i128;
    let unsigned = match tp {
        FieldType::Int(t) => !t.signed,
        _ => false,
    };
    (size, unsigned)
}

fn is_array_nolength(tp: &FieldType) -> bool {
    matches!(tp, FieldType::Array { .. })
}

/// `rffi_platform.py` `layout_addfield`.
fn layout_addfield(
    layout: &mut [Option<Rc<LayoutField>>],
    offset: usize,
    ctype: FieldType,
    prefix: &str,
) -> Rc<LayoutField> {
    let size = if is_array_nolength(&ctype) {
        layout.len() - offset
    } else {
        sizeof_type(&ctype)
    };
    // `while name in layout`: cells are Field or None, so a str never matches.
    let field = Rc::new(LayoutField {
        name: prefix.to_string(),
        ctype,
    });
    for slot in layout.iter_mut().skip(offset).take(size) {
        assert!(
            slot.is_none(),
            "{} overlaps {:?}",
            field.name,
            slot.as_ref().map(|f| &f.name)
        );
        *slot = Some(Rc::clone(&field));
    }
    field
}

/// `rffi_platform.py` `fixup_ctype` for a `Struct` field.
fn fixup_field_type(
    fieldtype: FieldType,
    fieldname: &str,
    size: i128,
    unsigned: bool,
) -> Result<FieldType, CompilationError> {
    match &fieldtype {
        FieldType::Int(_) => {
            for ctype in integer_class() {
                let u = !ctype.signed;
                let sz = (ctype.bits / 8) as i128;
                if (sz, u) == (size, unsigned) {
                    return Ok(FieldType::Int(ctype));
                }
            }
        }
        FieldType::Double => {
            if (size, unsigned) == (8, false) {
                return Ok(FieldType::Double);
            }
        }
        FieldType::FixedSizeArray { of, .. } => {
            let of_size = sizeof_type(of);
            let length = if of_size == 0 {
                0
            } else {
                (size as usize) / of_size
            };
            return Ok(FieldType::FixedSizeArray {
                of: of.clone(),
                length,
            });
        }
        FieldType::Array { .. } | FieldType::Nested { .. } | FieldType::Struct(_) => {
            return Ok(fieldtype);
        }
    }
    Err(CompilationError {
        out: String::new(),
        err: format!(
            "conflict between translating python and compiler field type for symbol {fieldname}, expected size+sign ({size}, {unsigned})"
        ),
    })
}

/// `Struct.build_result`.
fn struct_build_result(
    name: &str,
    interesting_fields: &[(String, FieldType)],
    ifdef: Option<&str>,
    info: &[(String, i128)],
    config_result: &ConfigResult<'_>,
) -> Result<ConfigValue, CompilationError> {
    if ifdef.is_some() && info_get(info, "defined") == 0 {
        return Ok(ConfigValue::None);
    }
    let size = info_get(info, "size") as usize;
    let mut layout: Vec<Option<Rc<LayoutField>>> = vec![None; size];
    for (fieldname, fieldtype) in interesting_fields {
        if let FieldType::Nested { c_name } = fieldtype {
            let offset = info_get(info, &format!("fldofs {fieldname}")) as usize;
            let c_fieldtype = config_result.get_struct_by_c_name(c_name)?;
            layout_addfield(
                &mut layout,
                offset,
                FieldType::Struct(c_fieldtype),
                fieldname,
            );
        } else {
            let offset = info_get(info, &format!("fldofs {fieldname}")) as usize;
            let fldsize = info_get(info, &format!("fldsize {fieldname}"));
            let sign = info_get_opt(info, &format!("fldunsigned {fieldname}")).unwrap_or(0) != 0;
            let mut fieldtype = fieldtype.clone();
            if is_array_nolength(&fieldtype) {
                // ignore size and sign
            } else if size_and_sign(&fieldtype) != (fldsize, sign) {
                fieldtype = fixup_field_type(fieldtype, fieldname, fldsize, sign)?;
            }
            layout_addfield(&mut layout, offset, fieldtype, fieldname);
        }
    }

    let mut n = 0;
    let mut padfields = Vec::new();
    for i in 0..layout.len() {
        if layout[i].is_some() {
            continue;
        }
        let pad_name = format!("_pad{n}");
        layout_addfield(&mut layout, i, FieldType::uchar(), &pad_name);
        padfields.push(format!("c_{pad_name}"));
        n += 1;
    }

    let mut seen: Vec<Rc<LayoutField>> = Vec::new();
    let mut fields = Vec::new();
    let mut fieldoffsets = Vec::new();
    for (offset, cell) in layout.iter().enumerate() {
        let cell = cell.as_ref().expect("layout filled");
        if seen.iter().any(|s| Rc::ptr_eq(s, cell)) {
            continue;
        }
        fields.push((format!("c_{}", cell.name), cell.ctype.clone()));
        fieldoffsets.push(offset as u64);
        seen.push(Rc::clone(cell));
    }

    let mut typedef = false;
    let stripped = if let Some(rest) = name.strip_prefix("struct ") {
        rest.to_string()
    } else {
        typedef = true;
        name.to_string()
    };
    Ok(ConfigValue::Struct(Rc::new(StructLayout {
        name: stripped,
        c_name: name.to_string(),
        typedef,
        align: info_get(info, "align") as u64,
        size: info_get(info, "size") as u64,
        fields,
        fieldoffsets,
        padding: padfields,
        eci: config_result.eci.clone(),
        padding_drop_cache: RefCell::new(None),
    })))
}

/// `PaddingDrop.compute_now`.
fn padding_drop_compute_now(
    name: &str,
    allfields: &[String],
    padfields: &[String],
    eci: &ExternalCompilationInfo,
    types: &[(String, String)],
) -> Result<Vec<String>, CompilationError> {
    let mut drops = Vec::new();
    let mut staticfields: Vec<Option<String>> = Vec::new();
    let mut consecutive_pads: Vec<String> = Vec::new();
    for fieldname in allfields {
        if padfields.iter().any(|p| p == fieldname) {
            consecutive_pads.push(fieldname.clone());
            continue;
        }
        let ctype = types
            .iter()
            .find(|(n, _)| n == fieldname)
            .map(|(_, t)| t.clone())
            .unwrap_or_else(|| panic!("KeyError: {fieldname}"));
        staticfields.push(Some(ctype));
        if !consecutive_pads.is_empty() {
            let mut found = None;
            for i in 0..=consecutive_pads.len() {
                let lookup = CConfigEntry::PaddingDropFieldLookup {
                    name: name.to_string(),
                    staticfields: staticfields.clone(),
                    fieldname: fieldname.clone(),
                };
                let cconfig = CConfig {
                    compilation_info: eci.clone(),
                    entries: vec![("FIELDLOOKUP".into(), lookup)],
                    single_entries: Vec::new(),
                };
                match configure(&cconfig, false) {
                    Ok(results) => {
                        let got = match &results[0].1 {
                            ConfigValue::Int(v) => *v,
                            other => panic!("FIELDLOOKUP: {other:?}"),
                        };
                        if got == 1 {
                            found = Some(i);
                            break;
                        }
                    }
                    Err(_) => {}
                }
                let last = staticfields.len() - 1;
                staticfields.insert(last, None);
            }
            let i = found.unwrap_or_else(|| {
                panic!("could not determine the detailed field layout of {name:?}")
            });
            drops.extend(consecutive_pads[i..].iter().cloned());
        }
        consecutive_pads.clear();
    }
    drops.extend(consecutive_pads);
    Ok(drops)
}

/// `rffi_platform.py` `CConfigSingleEntry`.
pub enum CConfigSingleEntry {
    /// `rffi_platform.py` `Has`.
    Has(String),
    /// `rffi_platform.py` `Works`.
    Works,
}

impl CConfigSingleEntry {
    fn question(
        &self,
        ask_gcc: &mut dyn FnMut(&str) -> Result<(), CompilationError>,
    ) -> Result<ConfigValue, CompilationError> {
        match self {
            CConfigSingleEntry::Has(name) => Ok(ConfigValue::Bool(
                ask_gcc(&format!("(void){name};")).is_ok(),
            )),
            CConfigSingleEntry::Works => {
                ask_gcc("")?;
                Ok(ConfigValue::None)
            }
        }
    }
}

/// `rffi_platform.py` `configure` result values.
#[derive(Clone, Debug, PartialEq)]
pub enum ConfigValue {
    Type(IntType),
    Size(u64),
    Bool(bool),
    Int(i128),
    Double(f64),
    Str(String),
    /// `Struct.build_result` / `rffi.CStruct`.
    Struct(Rc<StructLayout>),
    /// `#ifdef` false, an undefined `DefinedConstant*`, or `Works` success.
    None,
}

/// `rffi_platform.py` `ConfigResult`. Nested `Struct` fields call
/// `get_entry_result` on the inner entry (keyed here by C name).
struct ConfigResult<'a> {
    eci: ExternalCompilationInfo,
    entries: &'a [&'a CConfigEntry],
    info: Vec<Vec<(String, i128)>>,
    result: RefCell<Vec<Option<ConfigValue>>>,
}

impl<'a> ConfigResult<'a> {
    fn get_entry_result(&self, index: usize) -> Result<ConfigValue, CompilationError> {
        if let Some(v) = self.result.borrow()[index].clone() {
            return Ok(v);
        }
        let value = self.entries[index].build_result(&self.info[index], Some(self))?;
        self.result.borrow_mut()[index] = Some(value.clone());
        Ok(value)
    }

    fn get_struct_by_c_name(&self, c_name: &str) -> Result<Rc<StructLayout>, CompilationError> {
        for (i, entry) in self.entries.iter().enumerate() {
            if let CConfigEntry::Struct { name, .. } = entry {
                if name == c_name {
                    return match self.get_entry_result(i)? {
                        ConfigValue::Struct(s) => Ok(s),
                        ConfigValue::None => panic!("nested Struct {c_name} is None"),
                        other => panic!("nested Struct {c_name}: {other:?}"),
                    };
                }
            }
        }
        panic!("KeyError: nested Struct {c_name} is not a CConfigEntry");
    }
}

/// `rffi_platform.py` `CConfig`. Later attributes of the same name replace
/// earlier ones; `configure` sorts each list as `dir()` does.
pub struct CConfig {
    pub compilation_info: ExternalCompilationInfo,
    pub entries: Vec<(String, CConfigEntry)>,
    pub single_entries: Vec<(String, CConfigSingleEntry)>,
}

/// Collapse duplicate names (later wins) and sort as `dir()` does.
fn unique_sorted_by_name<T>(items: &[(String, T)]) -> Vec<(String, &T)> {
    let mut out: Vec<(String, &T)> = Vec::new();
    for (name, value) in items {
        if let Some(pos) = out.iter().position(|(n, _)| n == name) {
            out[pos].1 = value;
        } else {
            out.push((name.clone(), value));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `rffi_platform.py` `configure`. The result has one value per name, in
/// `dir()` order. `ignore_errors` is passed to `configure_entries` /
/// `run_example_code` / `gcc_cache.build_executable_cache`. `TARGET` and
/// `HOST` both set and different is a cross build: `platform()` is the
/// host (`host_triple` reads `HOST`), and only `host` and `arm`
/// (`set_platform` / `translationoption.PLATFORMS`) exist.
pub fn configure(
    cconfig: &CConfig,
    ignore_errors: bool,
) -> Result<Vec<(String, ConfigValue)>, CompilationError> {
    if let (Ok(target), Ok(host)) = (std::env::var("TARGET"), std::env::var("HOST"))
        && target != host
    {
        return Err(CompilationError {
            out: String::new(),
            err: format!(
                "rffi_platform.configure: TARGET {target} is not the host {host}; probing it needs a platform for {target} (set_platform), and only host and arm exist"
            ),
        });
    }
    let entries = unique_sorted_by_name(&cconfig.entries);
    let single_entries = unique_sorted_by_name(&cconfig.single_entries);
    for (name, _) in &entries {
        if single_entries.iter().any(|(n, _)| n == name) {
            panic!("CConfig attribute {name} is both a CConfigEntry and a CConfigSingleEntry");
        }
    }
    let mut res = Vec::new();
    if !entries.is_empty() {
        let list: Vec<&CConfigEntry> = entries.iter().map(|(_, e)| *e).collect();
        let results = configure_entries(&list, &cconfig.compilation_info, ignore_errors)?;
        for ((key, _), result) in entries.iter().zip(results) {
            res.push((key.clone(), result));
        }
    }
    for (key, entry) in &single_entries {
        let mut writer = CWriter::new(cconfig.compilation_info.clone());
        writer.write_header();
        let mut writer = Some(writer);
        let result = entry.question(&mut |q| writer.take().unwrap().ask_gcc(q))?;
        res.push((key.clone(), result));
    }
    Ok(res)
}

/// `rffi_platform.py` `configure_entries`. `ignore_errors` is
/// `run_example_code(..., ignore_errors=ignore_errors)`.
fn configure_entries(
    entries: &[&CConfigEntry],
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<Vec<ConfigValue>, CompilationError> {
    let mut writer = CWriter::new(eci.clone());
    writer.write_header();
    for (i, entry) in entries.iter().enumerate() {
        writer.write_entry(&i.to_string(), entry);
    }
    writer.start_main();
    for i in 0..entries.len() {
        writer.write_entry_main(&i.to_string());
    }
    writer.close()?;
    let infolist = run_example_code(&writer.path, eci, ignore_errors)?;
    assert_eq!(infolist.len(), entries.len());
    let config_result = ConfigResult {
        eci: eci.clone(),
        entries,
        info: infolist,
        result: RefCell::new(vec![None; entries.len()]),
    };
    let mut out = Vec::with_capacity(entries.len());
    for i in 0..entries.len() {
        out.push(config_result.get_entry_result(i)?);
    }
    Ok(out)
}

/// `rffi_platform.py` `_CWriter`.
struct CWriter {
    path: PathBuf,
    text: String,
    eci: ExternalCompilationInfo,
}

impl CWriter {
    fn new(eci: ExternalCompilationInfo) -> Self {
        CWriter {
            path: uniquefilepath(),
            text: String::new(),
            eci,
        }
    }

    fn write_header(&mut self) {
        self.eci.write_c_header(&mut self.text);
        self.text.push_str(C_HEADER);
        self.text.push('\n');
        self.text.push('\n');
    }

    fn write_entry(&mut self, key: &str, entry: &CConfigEntry) {
        self.text
            .push_str(&format!("void dump_section_{key}(void) {{\n"));
        for line in entry.prepare_code() {
            let line = if !line.is_empty() && !line.starts_with('#') {
                format!("\t{line}")
            } else {
                line
            };
            self.text.push_str(&line);
            self.text.push('\n');
        }
        self.text.push_str("}\n");
        self.text.push('\n');
    }

    fn write_entry_main(&mut self, key: &str) {
        self.text
            .push_str(&format!("\tprintf(\"-+- {key}\\n\");\n"));
        self.text.push_str(&format!("\tdump_section_{key}();\n"));
        self.text.push_str("\tprintf(\"---\\n\");\n");
    }

    fn start_main(&mut self) {
        self.text.push_str("int main(int argc, char *argv[]) {\n");
    }

    fn close(&mut self) -> Result<(), CompilationError> {
        self.text.push_str("\treturn 0;\n");
        self.text.push_str("}\n");
        fs::write(&self.path, &self.text).map_err(|err| CompilationError {
            out: String::new(),
            err: err.to_string(),
        })?;
        Ok(())
    }

    /// `rffi_platform.py` `_CWriter.ask_gcc`: `try_compile_cache([self.path], self.eci)`.
    fn ask_gcc(mut self, question: &str) -> Result<(), CompilationError> {
        self.start_main();
        self.text.push_str(question);
        self.text.push('\n');
        self.close()?;
        crate::gcc_cache::try_compile_cache(&[&self.path], &self.eci)
    }
}

/// `rffi_platform.py` `uniquefilepath(LAST=[0])`: `udir.join('platcheck_%d.c' % i)`.
fn uniquefilepath() -> PathBuf {
    static LAST: AtomicUsize = AtomicUsize::new(0);
    let i = LAST.fetch_add(1, Ordering::Relaxed);
    crate::udir::udir().join(format!("platcheck_{i}.c"))
}

/// `rffi_platform.py` `run_example_code`: `build_executable_cache(files, eci, ignore_errors=ignore_errors)` then parse stdout.
fn run_example_code(
    filepath: &Path,
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    let output = crate::gcc_cache::build_executable_cache(&[filepath], eci, ignore_errors)?;
    parse_example_output(&output)
}

/// `rffi_platform.py` `run_example_code` parsing loop.
fn parse_example_output(output: &str) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    if !output.starts_with("-+- ") {
        return Err(CompilationError {
            out: String::new(),
            err: "run_example_code failed!".into(),
        });
    }
    let mut sections = Vec::new();
    let mut section: Option<Vec<(String, i128)>> = None;
    for line in output.lines() {
        let line = line.trim();
        if line.starts_with("-+- ") {
            section = Some(Vec::new());
        } else if line == "---" {
            let done = section.take().expect("---");
            sections.push(done);
        } else if !line.is_empty() {
            let section = section.as_mut().expect("line");
            let (key, value) = line.split_once(": ").unwrap_or_else(|| panic!("{line}"));
            let value: i128 = value.parse().unwrap_or_else(|_| panic!("{value}"));
            if let Some((_, slot)) = section.iter_mut().find(|(k, _)| k == key) {
                *slot = value;
            } else {
                section.push((key.to_string(), value));
            }
        }
    }
    Ok(sections)
}

/// `rffi_platform.py` `eci_from_header`.
pub fn eci_from_header(
    c_header_source: impl Into<String>,
    include_dirs: Vec<String>,
    libraries: Vec<String>,
) -> ExternalCompilationInfo {
    ExternalCompilationInfo {
        post_include_bits: vec![c_header_source.into()],
        include_dirs,
        libraries,
        ..ExternalCompilationInfo::default()
    }
}

fn configure_one(
    eci: ExternalCompilationInfo,
    name: &str,
    entry: CConfigEntry,
) -> Result<ConfigValue, CompilationError> {
    let cconfig = CConfig {
        compilation_info: eci,
        entries: vec![(name.to_string(), entry)],
        single_entries: Vec::new(),
    };
    let mut results = configure(&cconfig, false)?;
    Ok(results.remove(0).1)
}

/// `rffi_platform.py` `getsimpletype`.
pub fn getsimpletype(name: &str, c_header_source: &str) -> Result<IntType, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "TYPE",
        CConfigEntry::simple_type(name),
    )? {
        ConfigValue::Type(t) => Ok(t),
        other => panic!("getsimpletype: {other:?}"),
    }
}

/// `rffi_platform.py` `getconstantinteger`.
pub fn getconstantinteger(name: &str, c_header_source: &str) -> Result<i128, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "CONST",
        CConfigEntry::ConstantInteger(name.into()),
    )? {
        ConfigValue::Int(v) => Ok(v),
        other => panic!("getconstantinteger: {other:?}"),
    }
}

/// `rffi_platform.py` `getdefined`.
pub fn getdefined(macro_name: &str, c_header_source: &str) -> Result<bool, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "DEFINED",
        CConfigEntry::Defined(macro_name.into()),
    )? {
        ConfigValue::Bool(v) => Ok(v),
        other => panic!("getdefined: {other:?}"),
    }
}

/// `rffi_platform.py` `getdefineddouble`.
pub fn getdefineddouble(
    macro_name: &str,
    c_header_source: &str,
) -> Result<Option<f64>, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "DEFINED",
        CConfigEntry::DefinedConstantDouble(macro_name.into()),
    )? {
        ConfigValue::Double(v) => Ok(Some(v)),
        ConfigValue::None => Ok(None),
        other => panic!("getdefineddouble: {other:?}"),
    }
}

/// `rffi_platform.py` `getdefinedinteger`.
pub fn getdefinedinteger(
    macro_name: &str,
    c_header_source: &str,
) -> Result<Option<i128>, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "DEFINED",
        CConfigEntry::DefinedConstantInteger(macro_name.into()),
    )? {
        ConfigValue::Int(v) => Ok(Some(v)),
        ConfigValue::None => Ok(None),
        other => panic!("getdefinedinteger: {other:?}"),
    }
}

/// `rffi_platform.py` `getdefinedstring`.
pub fn getdefinedstring(
    macro_name: &str,
    c_header_source: &str,
) -> Result<Option<String>, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "DEFINED",
        CConfigEntry::DefinedConstantString {
            macro_name: macro_name.into(),
            name: macro_name.into(),
        },
    )? {
        ConfigValue::Str(v) => Ok(Some(v)),
        ConfigValue::None => Ok(None),
        other => panic!("getdefinedstring: {other:?}"),
    }
}

/// `rffi_platform.py` `getintegerfunctionresult`.
pub fn getintegerfunctionresult(
    function: &str,
    args: &[&str],
    c_header_source: &str,
    includes: &[&str],
) -> Result<i128, CompilationError> {
    let mut eci = eci_from_header(c_header_source, Vec::new(), Vec::new());
    eci.includes = includes.iter().map(|s| (*s).to_string()).collect();
    match configure_one(
        eci,
        "RESULT",
        CConfigEntry::IntegerFunctionResult {
            name: function.into(),
            args: args.iter().map(|s| (*s).to_string()).collect(),
        },
    )? {
        ConfigValue::Int(v) => Ok(v),
        other => panic!("getintegerfunctionresult: {other:?}"),
    }
}

/// `rffi_platform.py` `has`.
pub fn has(
    name: &str,
    c_header_source: &str,
    include_dirs: Vec<String>,
    libraries: Vec<String>,
) -> Result<bool, CompilationError> {
    let cconfig = CConfig {
        compilation_info: eci_from_header(c_header_source, include_dirs, libraries),
        entries: Vec::new(),
        single_entries: vec![("HAS".into(), CConfigSingleEntry::Has(name.into()))],
    };
    match configure(&cconfig, false)?.remove(0).1 {
        ConfigValue::Bool(v) => Ok(v),
        other => panic!("has: {other:?}"),
    }
}

/// `rffi_platform.py` `verify_eci`.
pub fn verify_eci(eci: ExternalCompilationInfo) -> Result<(), CompilationError> {
    let cconfig = CConfig {
        compilation_info: eci,
        entries: Vec::new(),
        single_entries: vec![("WORKS".into(), CConfigSingleEntry::Works)],
    };
    configure(&cconfig, false).map(|_| ())
}

/// `rffi_platform.py` `checkcompiles`.
pub fn checkcompiles(
    expression: &str,
    c_header_source: &str,
    include_dirs: Vec<String>,
) -> Result<bool, CompilationError> {
    has(expression, c_header_source, include_dirs, Vec::new())
}

/// `rffi_platform.py` `sizeof`.
pub fn sizeof(name: &str, eci: ExternalCompilationInfo) -> Result<u64, CompilationError> {
    match configure_one(eci, "SIZE", CConfigEntry::SizeOf(name.into()))? {
        ConfigValue::Size(size) => Ok(size),
        other => panic!("sizeof: {other:?}"),
    }
}

/// `rffi_platform.py` `getstruct`.
pub fn getstruct(
    name: &str,
    c_header_source: &str,
    interesting_fields: Vec<(String, FieldType)>,
) -> Result<Rc<StructLayout>, CompilationError> {
    match configure_one(
        eci_from_header(c_header_source, Vec::new(), Vec::new()),
        "STRUCT",
        CConfigEntry::struct_type(name, interesting_fields, None),
    )? {
        ConfigValue::Struct(s) => Ok(s),
        other => panic!("getstruct: {other:?}"),
    }
}

/// `rffi_platform.py` `memory_alignment`.
pub fn memory_alignment() -> Result<u64, CompilationError> {
    static MEMORY_ALIGNMENT: OnceLock<u64> = OnceLock::new();
    if let Some(&v) = MEMORY_ALIGNMENT.get() {
        return Ok(v);
    }
    let result = memory_alignment_compute()?;
    Ok(*MEMORY_ALIGNMENT.get_or_init(|| result))
}

fn memory_alignment_compute() -> Result<u64, CompilationError> {
    if cfg!(windows) {
        Ok(std::mem::size_of::<usize>() as u64)
    } else {
        let s = getstruct(
            "struct memory_alignment_test",
            r#"
               struct memory_alignment_test {
                   double d;
                   void* p;
               };
            "#,
            Vec::new(),
        )?;
        let result = s.align;
        assert!(result & (result - 1) == 0, "not a power of two??");
        Ok(result)
    }
}

/// `rffi_platform.py` `configure_external_library` configuration dict.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LibraryConfiguration {
    pub prefix: String,
    pub include_dir: String,
    pub library_dir: String,
}

/// `rffi_platform.py` `PYPY_EXTERNAL_DIR`.
fn pypy_external_dir() -> PathBuf {
    let rpython_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("rpython");
    let mut dir = rpython_dir.join("..").join("..");
    if cfg!(windows) {
        for libdir in [
            PathBuf::from(r"c:\buildslave\support"),
            PathBuf::from(r"d:\myslave"),
        ] {
            if libdir.is_dir() {
                dir = libdir;
                break;
            }
        }
    }
    dir
}

fn external_library_cache()
-> &'static Mutex<Vec<((String, ExternalCompilationInfo), ExternalCompilationInfo)>> {
    static CACHE: OnceLock<
        Mutex<Vec<((String, ExternalCompilationInfo), ExternalCompilationInfo)>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn glob_prefix(dir: &Path, prefix: &str) -> Vec<PathBuf> {
    let mut entries = Vec::new();
    let Ok(read) = fs::read_dir(dir) else {
        return entries;
    };
    for ent in read.flatten() {
        let name = ent.file_name();
        if name.to_string_lossy().starts_with(prefix) {
            entries.push(ent.path());
        }
    }
    entries.sort();
    entries
}

/// `rffi_platform.py` `configure_external_library`. `symbol` is unused
/// upstream (`configure_boehm` still passes `symbol='GC_init'`).
pub fn configure_external_library(
    name: &str,
    eci: ExternalCompilationInfo,
    mut configurations: Vec<LibraryConfiguration>,
    _symbol: Option<&str>,
) -> Result<ExternalCompilationInfo, CompilationError> {
    if !cfg!(windows) {
        configurations = Vec::new();
    }
    {
        let cache = external_library_cache().lock();
        if let Some((_, eci_lib)) = cache
            .iter()
            .find(|((n, cached), _)| n == name && cached == &eci)
        {
            return Ok(eci_lib.clone());
        }
    }

    let mut last_error = None;
    if !configurations.contains(&LibraryConfiguration::default()) {
        configurations.push(LibraryConfiguration::default());
    }

    for configuration in configurations {
        let mut prefix = configuration.prefix;
        let include_dir = configuration.include_dir;
        let library_dir = configuration.library_dir;

        if !prefix.is_empty() && !Path::new(&prefix).is_absolute() {
            let entries = glob_prefix(&pypy_external_dir(), &prefix);
            if let Some(last) = entries.last() {
                prefix = last.to_string_lossy().into_owned();
            } else {
                continue;
            }
        }

        let include_dir = Path::new(&prefix)
            .join(&include_dir)
            .to_string_lossy()
            .into_owned();
        let library_dir = Path::new(&prefix)
            .join(&library_dir)
            .to_string_lossy()
            .into_owned();

        let eci_lib = ExternalCompilationInfo {
            include_dirs: if include_dir.is_empty() {
                Vec::new()
            } else {
                vec![include_dir]
            },
            library_dirs: if library_dir.is_empty() {
                Vec::new()
            } else {
                vec![library_dir]
            },
            ..ExternalCompilationInfo::default()
        };
        let eci_lib = eci_lib.merge(&[&eci]);

        match verify_eci(eci_lib.clone()) {
            Ok(()) => {
                external_library_cache()
                    .lock()
                    .push(((name.to_string(), eci), eci_lib.clone()));
                return Ok(eci_lib);
            }
            Err(e) => last_error = Some(e),
        }
    }

    if let Some(last_error) = last_error {
        Err(last_error)
    } else {
        Err(CompilationError {
            out: String::new(),
            err: format!("Library {name} is not installed"),
        })
    }
}

/// `rffi_platform.py` `configure_boehm`.
pub fn configure_boehm(
    platform: Option<String>,
) -> Result<ExternalCompilationInfo, CompilationError> {
    let (library_dir, libraries, includes, platform) = if cfg!(windows) {
        let (library_dir, libraries) = if usize::BITS == 32 {
            ("Release", vec!["gc".to_string()])
        } else {
            ("", vec!["gc64_dll".to_string()])
        };
        (library_dir, libraries, vec!["gc.h".to_string()], None)
    } else {
        (
            "",
            vec!["gc".to_string()],
            vec!["gc/gc.h".to_string()],
            platform,
        )
    };
    match ExternalCompilationInfo::from_pkg_config("bdw-gc") {
        Ok(mut eci) => {
            eci.includes.extend(includes);
            return Ok(eci);
        }
        Err(_) => {}
    }
    let eci = ExternalCompilationInfo {
        platform,
        includes,
        libraries,
        ..ExternalCompilationInfo::default()
    };
    configure_external_library(
        "gc",
        eci,
        vec![LibraryConfiguration {
            prefix: "gc-".into(),
            include_dir: "include".into(),
            library_dir: library_dir.into(),
        }],
        Some("GC_init"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rffi_platform_entries() {
        assert_eq!(
            CConfigEntry::simple_type("mode_t").prepare_code(),
            vec![
                "typedef mode_t platcheck_t;".to_string(),
                String::new(),
                "platcheck_t x;".to_string(),
                "dump(\"size\",  sizeof(platcheck_t));".to_string(),
                "x = 0; x = ~x;".to_string(),
                "dump(\"unsigned\", x > 0);".to_string(),
            ]
        );
        assert_eq!(
            CConfigEntry::SizeOf("long double".into()).prepare_code(),
            vec!["dump(\"size\",  sizeof(long double));".to_string()]
        );
        assert_eq!(
            CConfigEntry::ConstantInteger("EOF".into()).prepare_code(),
            vec![
                "if ((EOF) <= 0) {".to_string(),
                "    long long x = (long long)(EOF);".to_string(),
                "    printf(\"value: %lld\\n\", x);".to_string(),
                "} else {".to_string(),
                "    unsigned long long x = (unsigned long long)(EOF);".to_string(),
                "    printf(\"value: %llu\\n\", x);".to_string(),
                "}".to_string(),
            ]
        );

        let mut writer = CWriter::new(ExternalCompilationInfo::default());
        writer.write_header();
        writer.write_entry("0", &CConfigEntry::SizeOf("int".into()));
        writer.write_entry("1", &CConfigEntry::SizeOf("long".into()));
        writer.start_main();
        writer.write_entry_main("0");
        writer.write_entry_main("1");
        writer.close().unwrap();
        assert!(
            writer.text.contains("void dump_section_0(void) {"),
            "{}",
            writer.text
        );
        assert!(
            writer.text.contains("\tdump_section_1();"),
            "{}",
            writer.text
        );
        assert!(writer.text.ends_with("\treturn 0;\n}\n"), "{}", writer.text);

        let parsed =
            parse_example_output("-+- 0\nsize: 4\nunsigned: 1\n---\n-+- 1\nsize: 16\n---\n")
                .unwrap();
        assert_eq!(
            parsed,
            vec![
                vec![("size".into(), 4), ("unsigned".into(), 1)],
                vec![("size".into(), 16)],
            ]
        );
        let parsed_dup = parse_example_output("-+- 0\nsize: 4\nsize: 8\n---\n").unwrap();
        assert_eq!(parsed_dup, vec![vec![("size".into(), 8)]]);
        assert!(parse_example_output("oops").is_err());

        let st = CConfigEntry::simple_type("mode_t");
        assert_eq!(
            st.build_result(&[("size".into(), 4), ("unsigned".into(), 1)], None)
                .unwrap(),
            ConfigValue::Type(IntType {
                signed: false,
                bits: 32
            })
        );
        assert_eq!(
            st.build_result(&[("size".into(), 8), ("unsigned".into(), 0)], None)
                .unwrap(),
            ConfigValue::Type(IntType {
                signed: true,
                bits: 64
            })
        );
        assert!(
            st.build_result(&[("size".into(), 16), ("unsigned".into(), 0)], None)
                .is_err()
        );

        #[cfg(unix)]
        {
            // `uniquefilepath` uses `udir()`; a leftover `PYPY_USESSION_DIR`
            // must not be required for gcc_cache to compile the probe.
            let previous_session = std::env::var_os("PYPY_USESSION_DIR");
            // SAFETY: this test is the only code touching `PYPY_USESSION_DIR`
            // around `configure`; `udir()` is a process-wide OnceLock.
            unsafe {
                std::env::remove_var("PYPY_USESSION_DIR");
            }
            let cconfig = CConfig {
                compilation_info: ExternalCompilationInfo {
                    includes: vec!["stdio.h".into()],
                    ..ExternalCompilationInfo::default()
                },
                entries: vec![
                    ("SIZE".into(), CConfigEntry::SizeOf("int".into())),
                    ("TYPE".into(), CConfigEntry::simple_type("unsigned short")),
                    ("EOF".into(), CConfigEntry::ConstantInteger("EOF".into())),
                ],
                single_entries: vec![
                    ("HAS".into(), CConfigSingleEntry::Has("printf".into())),
                    (
                        "NO".into(),
                        CConfigSingleEntry::Has("no_such_function_xyz".into()),
                    ),
                ],
            };
            let results = configure(&cconfig, false).unwrap();
            match previous_session {
                Some(value) => unsafe { std::env::set_var("PYPY_USESSION_DIR", value) },
                None => unsafe { std::env::remove_var("PYPY_USESSION_DIR") },
            }
            assert_eq!(
                results,
                vec![
                    ("EOF".into(), ConfigValue::Int(-1)),
                    ("SIZE".into(), ConfigValue::Size(4)),
                    (
                        "TYPE".into(),
                        ConfigValue::Type(IntType {
                            signed: false,
                            bits: 16
                        })
                    ),
                    ("HAS".into(), ConfigValue::Bool(true)),
                    ("NO".into(), ConfigValue::Bool(false)),
                ]
            );
        }
    }

    #[test]
    #[should_panic(
        expected = "CConfig attribute SIZE is both a CConfigEntry and a CConfigSingleEntry"
    )]
    fn cconfig_name_in_both_lists_panics() {
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo::default(),
            entries: vec![("SIZE".into(), CConfigEntry::SizeOf("int".into()))],
            single_entries: vec![("SIZE".into(), CConfigSingleEntry::Has("printf".into()))],
        };
        let _ = configure(&cconfig, false);
    }

    #[test]
    fn test_simple_type() {
        let ctype = getsimpletype("test_t", "typedef unsigned short test_t;").unwrap();
        assert_eq!(
            ctype,
            IntType {
                signed: false,
                bits: 16
            }
        );
    }

    #[test]
    fn test_constant_integer() {
        assert_eq!(
            getconstantinteger("BLAH", "#define BLAH (6*7)").unwrap(),
            42
        );
        assert_eq!(
            getconstantinteger("BLAH", "#define BLAH (-2147483648LL)").unwrap(),
            -2147483648
        );
        assert_eq!(
            getconstantinteger("BLAH", "#define BLAH (3333333333ULL)").unwrap(),
            3333333333
        );
    }

    #[test]
    fn test_defined() {
        assert!(!getdefined("ALFKJLKJFLKJFKLEJDLKEWMECEE", "").unwrap());
        assert!(
            getdefined(
                "ALFKJLKJFLKJFKLEJDLKEWMECEE",
                "#define ALFKJLKJFLKJFKLEJDLKEWMECEE"
            )
            .unwrap()
        );
    }

    #[test]
    fn test_defined_constant() {
        assert_eq!(
            getdefineddouble("ABCDFGH", "#define ABCDFGH 2.0").unwrap(),
            Some(2.0)
        );
        assert_eq!(
            getdefinedinteger("ABCDFGH", "#define ABCDFGH 2").unwrap(),
            Some(2)
        );
    }

    #[test]
    fn test_defined_constant_float() {
        assert_eq!(
            getdefineddouble("BLAH", "#define BLAH 1.0").unwrap(),
            Some(1.0)
        );
        assert_eq!(
            getdefineddouble("BLAH", "#define BLAH 1.5").unwrap(),
            Some(1.5)
        );
        assert_eq!(
            getdefineddouble("BLAH", "#define BLAH 1.0e20").unwrap(),
            Some(1.0e20)
        );
        #[cfg(not(target_env = "msvc"))]
        {
            let inf = getdefineddouble("BLAH", "#define BLAH 1.0e50000")
                .unwrap()
                .unwrap();
            assert!(inf.is_infinite() && inf.is_sign_positive(), "{inf}");
            let nan = getdefineddouble("BLAH", "#define BLAH (double)0/0")
                .unwrap()
                .unwrap();
            assert!(nan.is_nan(), "{nan}");
        }
    }

    #[test]
    fn test_defined_constant_string() {
        assert_eq!(getdefinedstring("MCDONC", "").unwrap(), None);
        assert_eq!(
            getdefinedstring("RAYDEO", "#define RAYDEO \"Michael Merickel\"").unwrap(),
            Some("Michael Merickel".into())
        );
    }

    #[test]
    fn test_getintegerfunctionresult() {
        let func = "RPY_EXPORTED int sum(int a, int b) {return a + b;}";
        assert_eq!(
            getintegerfunctionresult("sum", &["6", "7"], func, &[]).unwrap(),
            13
        );
        #[cfg(not(target_env = "msvc"))]
        {
            assert_eq!(
                getintegerfunctionresult("lround", &["6.7"], "#include <math.h>", &[]).unwrap(),
                7
            );
            assert_eq!(
                getintegerfunctionresult("lround", &["9.1"], "", &["math.h"]).unwrap(),
                9
            );
        }
    }

    #[test]
    fn test_integer_function_result() {
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo {
                pre_include_bits: vec!["int sum(int a, int b){ return a+b;}".into()],
                ..ExternalCompilationInfo::default()
            },
            entries: vec![
                (
                    "SUM".into(),
                    CConfigEntry::IntegerFunctionResult {
                        name: "sum".into(),
                        args: vec!["12".into(), "34".into()],
                    },
                ),
                (
                    "SUM2".into(),
                    CConfigEntry::IntegerFunctionResult {
                        name: "sum".into(),
                        args: vec!["-12".into(), "-34".into()],
                    },
                ),
            ],
            single_entries: Vec::new(),
        };
        let results = configure(&cconfig, false).unwrap();
        assert_eq!(
            results,
            vec![
                ("SUM".into(), ConfigValue::Int(46)),
                ("SUM2".into(), ConfigValue::Int(-46)),
            ]
        );
    }

    #[test]
    fn test_ifdef() {
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo {
                post_include_bits: vec![
                    "/* a C comment */".into(),
                    "#define XYZZY 42".into(),
                    "typedef int foo;".into(),
                    r#"
                                 struct s {
                                   int i;
                                   double f;
                                 };
                                 "#
                    .into(),
                ],
                ..ExternalCompilationInfo::default()
            },
            entries: vec![
                (
                    "s".into(),
                    CConfigEntry::struct_type(
                        "struct s",
                        vec![("i".into(), FieldType::int())],
                        Some("XYZZY".into()),
                    ),
                ),
                (
                    "z".into(),
                    CConfigEntry::struct_type(
                        "struct z",
                        vec![("i".into(), FieldType::int())],
                        Some("FOOBAR".into()),
                    ),
                ),
                (
                    "foo".into(),
                    CConfigEntry::SimpleType {
                        name: "foo".into(),
                        ifdef: Some("XYZZY".into()),
                    },
                ),
                (
                    "bar".into(),
                    CConfigEntry::SimpleType {
                        name: "bar".into(),
                        ifdef: Some("FOOBAR".into()),
                    },
                ),
            ],
            single_entries: Vec::new(),
        };
        let results = configure(&cconfig, false).unwrap();
        let get = |name: &str| {
            results
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v)
                .unwrap()
        };
        assert!(matches!(get("s"), ConfigValue::Struct(_)));
        assert_eq!(get("z"), &ConfigValue::None);
        assert_eq!(
            get("foo"),
            &ConfigValue::Type(IntType {
                signed: true,
                bits: 32
            })
        );
        assert_eq!(get("bar"), &ConfigValue::None);
    }

    #[test]
    fn test_has() {
        assert!(has("x", "int x = 3;", Vec::new(), Vec::new()).unwrap());
        assert!(!has("x", "", Vec::new(), Vec::new()).unwrap());
        assert!(
            !has(
                "x",
                "#include <some/path/which/cannot/exist>",
                Vec::new(),
                Vec::new()
            )
            .unwrap()
        );
    }

    #[test]
    fn test_verify_eci() {
        verify_eci(ExternalCompilationInfo::default()).unwrap();
        assert!(
            verify_eci(ExternalCompilationInfo {
                libraries: vec!["some_name_that_doesnt_exist_".into()],
                ..ExternalCompilationInfo::default()
            })
            .is_err()
        );
    }

    #[test]
    fn test_sizeof() {
        assert_eq!(
            sizeof("char", ExternalCompilationInfo::default()).unwrap(),
            1
        );
    }

    fn assert_int(tp: &FieldType, signed: bool, bits: u32) {
        match tp {
            FieldType::Int(t) => assert_eq!(
                *t,
                IntType { signed, bits },
                "expected signed={signed} bits={bits}"
            ),
            other => panic!("expected Int, got {other:?}"),
        }
    }

    #[test]
    fn test_dirent() {
        let header = r#"
           struct dirent  /* for this example only, not the exact dirent */
           {
               long d_ino;
               int d_off;
               unsigned short d_reclen;
               char d_name[32];
           };
                                       "#;
        let dirent = getstruct(
            "struct dirent",
            header,
            vec![("d_reclen".into(), FieldType::ushort())],
        )
        .unwrap();
        assert_int(dirent.c_field("c_d_reclen"), false, 16);
        let expected = sizeof(
            "struct dirent",
            eci_from_header(header, Vec::new(), Vec::new()),
        )
        .unwrap();
        assert_eq!(dirent.size, expected);
        assert!(!dirent.typedef);
        assert_eq!(dirent.name, "dirent");
    }

    #[test]
    fn test_fit_type() {
        let s = getstruct(
            "struct S",
            r#"
           struct S {
               signed char c;
               unsigned char uc;
               short s;
               unsigned short us;
               int i;
               unsigned int ui;
               long l;
               unsigned long ul;
               long long ll;
               unsigned long long ull;
               double d;
           };
                                  "#,
            vec![
                ("c".into(), FieldType::int()),
                ("uc".into(), FieldType::int()),
                ("s".into(), FieldType::uint()),
                ("us".into(), FieldType::int()),
                ("i".into(), FieldType::int()),
                ("ui".into(), FieldType::int()),
                ("l".into(), FieldType::int()),
                ("ul".into(), FieldType::int()),
                ("ll".into(), FieldType::int()),
                ("ull".into(), FieldType::int()),
                ("d".into(), FieldType::double()),
            ],
        )
        .unwrap();
        assert_int(s.c_field("c_c"), true, 8);
        assert_int(s.c_field("c_uc"), false, 8);
        assert_int(s.c_field("c_s"), true, 16);
        assert_int(s.c_field("c_us"), false, 16);
        assert_int(s.c_field("c_i"), true, 32);
        assert_int(s.c_field("c_ui"), false, 32);
        let long_bits = if cfg!(windows) {
            32
        } else {
            (std::mem::size_of::<usize>() * 8) as u32
        };
        assert_int(s.c_field("c_l"), true, long_bits);
        assert_int(s.c_field("c_ul"), false, long_bits);
        assert_int(s.c_field("c_ll"), true, 64);
        assert_int(s.c_field("c_ull"), false, 64);
        assert_eq!(s.c_field("c_d"), &FieldType::Double);
    }

    #[test]
    fn test_configure() {
        let header = crate::udir::udir().join("test_ctypes_platform.h");
        fs::write(&header, "#define XYZZY 42\n").unwrap();
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo {
                pre_include_bits: vec![
                    "/* a C comment */".into(),
                    "#include <stdio.h>".into(),
                    "#include <test_ctypes_platform.h>".into(),
                ],
                include_dirs: vec![crate::udir::udir().display().to_string()],
                ..ExternalCompilationInfo::default()
            },
            entries: vec![
                (
                    "FILE".into(),
                    CConfigEntry::struct_type("FILE", Vec::new(), None),
                ),
                ("ushort".into(), CConfigEntry::simple_type("unsigned short")),
                (
                    "XYZZY".into(),
                    CConfigEntry::ConstantInteger("XYZZY".into()),
                ),
            ],
            single_entries: Vec::new(),
        };
        let results = configure(&cconfig, false).unwrap();
        let get = |name: &str| {
            results
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v)
                .unwrap()
        };
        assert!(matches!(get("FILE"), ConfigValue::Struct(_)));
        assert_eq!(
            get("ushort"),
            &ConfigValue::Type(IntType {
                signed: false,
                bits: 16
            })
        );
        assert_eq!(get("XYZZY"), &ConfigValue::Int(42));

        #[cfg(unix)]
        {
            let timeval_cconfig = CConfig {
                compilation_info: ExternalCompilationInfo {
                    includes: vec!["sys/time.h".into()],
                    ..ExternalCompilationInfo::default()
                },
                entries: vec![(
                    "TIMEVAL".into(),
                    CConfigEntry::struct_type(
                        "struct timeval",
                        vec![
                            ("tv_sec".into(), FieldType::int()),
                            ("tv_usec".into(), FieldType::int()),
                        ],
                        None,
                    ),
                )],
                single_entries: Vec::new(),
            };
            let timeval_results = configure(&timeval_cconfig, false).unwrap();
            let layout = match &timeval_results[0].1 {
                ConfigValue::Struct(s) => s,
                other => panic!("{other:?}"),
            };
            let src = layout.rust_struct_source("TIMEVAL", &[]);
            assert!(src.contains("pub c_tv_sec:"), "{src}");
            assert!(
                src.contains(&format!(
                    "const _: () = assert!(::core::mem::size_of::<TIMEVAL>() == {} && ::core::mem::align_of::<TIMEVAL>() == {});",
                    layout.size, layout.align
                )),
                "{src}"
            );
        }
    }

    fn nested_xy_header(inner_first: bool) -> String {
        if inner_first {
            r#"
            struct x {
            int foo;
            unsigned long bar;
            };
            struct y {
            char c;
            struct x x;
            };
            "#
            .into()
        } else {
            r#"
            struct y {
            int foo;
            unsigned long bar;
            };
            struct x {
            char c;
            struct y y;
            };
            "#
            .into()
        }
    }

    #[test]
    fn test_nested_structs() {
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo {
                post_include_bits: vec![nested_xy_header(true)],
                ..ExternalCompilationInfo::default()
            },
            entries: vec![
                (
                    "x".into(),
                    CConfigEntry::struct_type(
                        "struct x",
                        vec![(
                            "bar".into(),
                            FieldType::Int(IntType {
                                signed: true,
                                bits: 16,
                            }),
                        )],
                        None,
                    ),
                ),
                (
                    "y".into(),
                    CConfigEntry::struct_type(
                        "struct y",
                        vec![("x".into(), FieldType::nested("struct x"))],
                        None,
                    ),
                ),
            ],
            single_entries: Vec::new(),
        };
        let results = configure(&cconfig, false).unwrap();
        let c_x = match &results.iter().find(|(n, _)| n == "x").unwrap().1 {
            ConfigValue::Struct(s) => s,
            other => panic!("{other:?}"),
        };
        let c_y = match &results.iter().find(|(n, _)| n == "y").unwrap().1 {
            ConfigValue::Struct(s) => s,
            other => panic!("{other:?}"),
        };
        match c_y.c_field("c_x") {
            FieldType::Struct(inner) => assert!(Rc::ptr_eq(inner, c_x)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_nested_structs_in_the_opposite_order() {
        let cconfig = CConfig {
            compilation_info: ExternalCompilationInfo {
                post_include_bits: vec![nested_xy_header(false)],
                ..ExternalCompilationInfo::default()
            },
            entries: vec![
                (
                    "y".into(),
                    CConfigEntry::struct_type(
                        "struct y",
                        vec![(
                            "bar".into(),
                            FieldType::Int(IntType {
                                signed: true,
                                bits: 16,
                            }),
                        )],
                        None,
                    ),
                ),
                (
                    "x".into(),
                    CConfigEntry::struct_type(
                        "struct x",
                        vec![("y".into(), FieldType::nested("struct y"))],
                        None,
                    ),
                ),
            ],
            single_entries: Vec::new(),
        };
        let results = configure(&cconfig, false).unwrap();
        let c_x = match &results.iter().find(|(n, _)| n == "x").unwrap().1 {
            ConfigValue::Struct(s) => s,
            other => panic!("{other:?}"),
        };
        let c_y = match &results.iter().find(|(n, _)| n == "y").unwrap().1 {
            ConfigValue::Struct(s) => s,
            other => panic!("{other:?}"),
        };
        match c_x.c_field("c_y") {
            FieldType::Struct(inner) => assert!(Rc::ptr_eq(inner, c_y)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_array() {
        let dirent = getstruct(
            "struct dirent",
            r#"
           struct dirent  /* for this example only, not the exact dirent */
           {
               long d_ino;
               int d_off;
               unsigned short d_reclen;
               char d_name[32];
           };
                                       "#,
            vec![(
                "d_name".into(),
                FieldType::fixed_array(FieldType::char(), 1),
            )],
        )
        .unwrap();
        match dirent.c_field("c_d_name") {
            FieldType::FixedSizeArray { length, .. } => assert_eq!(*length, 32),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_array_varsized_struct() {
        let dirent = getstruct(
            "struct dirent",
            r#"
           struct dirent  /* for this example only, not the exact dirent */
           {
               int d_off;
               char d_name[1];
           };
                                       "#,
            vec![("d_name".into(), FieldType::array(FieldType::char()))],
        )
        .unwrap();
        assert_eq!(dirent.offsetof("c_d_name"), 4);
        match dirent.c_field("c_d_name") {
            FieldType::Array { of } => assert_eq!(of.as_ref(), &FieldType::char()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_memory_alignment() {
        let a = memory_alignment().unwrap();
        let pointer = std::mem::size_of::<usize>() as u64;
        assert_eq!(a % pointer, 0, "{a}");
    }

    fn padding_types(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(n, t)| ((*n).to_string(), (*t).to_string()))
            .collect()
    }

    #[test]
    fn test_generate_padding() {
        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;        /* followed by one byte of padding */
                short s1;
           } foobar_t;
           "#,
            vec![
                ("c1".into(), FieldType::signed_word()),
                ("s1".into(), FieldType::signed_word()),
            ],
        )
        .unwrap();
        assert_eq!(s.padding, vec!["c__pad0".to_string()]);
        assert_eq!(
            s.get_padding_drop(&padding_types(&[("c_c1", "char"), ("c_s1", "short")]))
                .unwrap(),
            vec!["c__pad0".to_string()]
        );

        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;
                char c2;  /* _pad0 */
                short s1;
           } foobar_t;
           "#,
            vec![
                ("c1".into(), FieldType::signed_word()),
                ("s1".into(), FieldType::signed_word()),
            ],
        )
        .unwrap();
        assert_eq!(s.padding, vec!["c__pad0".to_string()]);
        assert_eq!(
            s.get_padding_drop(&padding_types(&[("c_c1", "char"), ("c_s1", "short")]))
                .unwrap(),
            Vec::<String>::new()
        );

        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;
                char c2;  /* _pad0 */
                /* _pad1, _pad2 */
                int i1;
           } foobar_t;
           "#,
            vec![
                ("c1".into(), FieldType::signed_word()),
                ("i1".into(), FieldType::signed_word()),
            ],
        )
        .unwrap();
        assert_eq!(
            s.padding,
            vec![
                "c__pad0".to_string(),
                "c__pad1".to_string(),
                "c__pad2".to_string()
            ]
        );
        assert_eq!(
            s.get_padding_drop(&padding_types(&[("c_c1", "char"), ("c_i1", "int")]))
                .unwrap(),
            vec!["c__pad1".to_string(), "c__pad2".to_string()]
        );

        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;
                char c2;  /* _pad0 */
                char c3;  /* _pad1 */
                /* _pad2 */
                int i1;
           } foobar_t;
           "#,
            vec![
                ("c1".into(), FieldType::signed_word()),
                ("i1".into(), FieldType::signed_word()),
            ],
        )
        .unwrap();
        assert_eq!(
            s.padding,
            vec![
                "c__pad0".to_string(),
                "c__pad1".to_string(),
                "c__pad2".to_string()
            ]
        );
        assert_eq!(
            s.get_padding_drop(&padding_types(&[("c_c1", "char"), ("c_i1", "int")]))
                .unwrap(),
            vec!["c__pad2".to_string()]
        );

        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;
                /* _pad0 */
                short s1;  /* _pad1, _pad2 */
                int i1;
           } foobar_t;
           "#,
            vec![
                ("c1".into(), FieldType::signed_word()),
                ("i1".into(), FieldType::signed_word()),
            ],
        )
        .unwrap();
        assert_eq!(
            s.padding,
            vec![
                "c__pad0".to_string(),
                "c__pad1".to_string(),
                "c__pad2".to_string()
            ]
        );
        assert_eq!(
            s.get_padding_drop(&padding_types(&[("c_c1", "char"), ("c_i1", "int")]))
                .unwrap(),
            vec!["c__pad1".to_string(), "c__pad2".to_string()]
        );

        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;
                char c2;  /* _pad0 */
                /* _pad1, _pad2 */
                int i1;
                char c3;  /* _pad3 */
                /* _pad4 */
                short s1;
           } foobar_t;
           "#,
            vec![
                ("c1".into(), FieldType::signed_word()),
                ("i1".into(), FieldType::signed_word()),
                ("s1".into(), FieldType::signed_word()),
            ],
        )
        .unwrap();
        assert_eq!(
            s.padding,
            vec![
                "c__pad0".to_string(),
                "c__pad1".to_string(),
                "c__pad2".to_string(),
                "c__pad3".to_string(),
                "c__pad4".to_string()
            ]
        );
        assert_eq!(
            s.get_padding_drop(&padding_types(&[
                ("c_c1", "char"),
                ("c_i1", "int"),
                ("c_s1", "short")
            ]))
            .unwrap(),
            vec![
                "c__pad1".to_string(),
                "c__pad2".to_string(),
                "c__pad4".to_string()
            ]
        );

        let s = getstruct(
            "foobar_t",
            r#"
           typedef struct {
                char c1;
                long l2;  /* some number of _pads */
           } foobar_t;
           "#,
            vec![("c1".into(), FieldType::signed_word())],
        )
        .unwrap();
        let padding = s.padding.clone();
        assert_eq!(
            s.get_padding_drop(&padding_types(&[("c_c1", "char")]))
                .unwrap(),
            padding
        );
    }

    #[test]
    fn test_configure_external_library_default() {
        let eci =
            configure_external_library("c", ExternalCompilationInfo::default(), Vec::new(), None)
                .unwrap();
        assert_eq!(eci, ExternalCompilationInfo::default());
        let again =
            configure_external_library("c", ExternalCompilationInfo::default(), Vec::new(), None)
                .unwrap();
        assert_eq!(eci, again);
    }

    #[test]
    fn test_configure_external_library_missing() {
        let err = configure_external_library(
            "not_a_real_lib_xyz",
            ExternalCompilationInfo {
                libraries: vec!["not_a_real_lib_xyz".into()],
                ..ExternalCompilationInfo::default()
            },
            Vec::new(),
            None,
        )
        .unwrap_err();
        assert!(!err.out.is_empty() || !err.err.is_empty(), "{err}");
    }

    #[test]
    fn test_configure_boehm() {
        match configure_boehm(None) {
            Ok(eci) => {
                assert!(
                    eci.libraries.iter().any(|lib| lib.contains("gc"))
                        || eci.includes.iter().any(|inc| inc.contains("gc")),
                    "{eci:?}"
                );
            }
            Err(_) => {}
        }
    }
}
