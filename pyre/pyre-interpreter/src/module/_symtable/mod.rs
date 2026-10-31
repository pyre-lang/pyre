//! `_symtable` — compiler symbol-table accelerator.
//!
//! PyPy exposes its compiler's `SymtableEntry` objects directly.  Pyre's
//! compiler is the shared RustPython compiler, so the native half converts its
//! `SymbolTable` tree into GC-visible Python containers; the small app-level
//! `_Entry` wrapper then presents the CPython/PyPy attribute surface consumed
//! by `Lib/symtable.py`.  No Rust side-table owns entry identity or lifetime.

use pyre_object::PyObjectRef;
use rustpython_compiler::codegen::symboltable::{
    CompilerScope, SymbolFlags, SymbolScope, SymbolTable,
};

const SCOPE_OFF: i32 = 12;
const TYPE_FUNCTION: i32 = 0;
const TYPE_CLASS: i32 = 1;
const TYPE_MODULE: i32 = 2;
const TYPE_ANNOTATION: i32 = 3;
const TYPE_TYPE_ALIAS: i32 = 4;
const TYPE_TYPE_PARAMETERS: i32 = 5;
const TYPE_TYPE_VARIABLE: i32 = 6;
const PUBLIC_SYMBOL_FLAGS: SymbolFlags = SymbolFlags::from_bits_retain(
    SymbolFlags::DEF_GLOBAL.bits()
        | SymbolFlags::DEF_LOCAL.bits()
        | SymbolFlags::DEF_PARAM.bits()
        | SymbolFlags::DEF_NONLOCAL.bits()
        | SymbolFlags::USE.bits()
        | SymbolFlags::DEF_FREE_CLASS.bits()
        | SymbolFlags::DEF_IMPORT.bits()
        | SymbolFlags::DEF_ANNOT.bits()
        | SymbolFlags::DEF_COMP_ITER.bits()
        | SymbolFlags::DEF_TYPE_PARAM.bits()
        | SymbolFlags::DEF_COMP_CELL.bits(),
);
/// `pycore_symtable.h DEF_BOUND` — the three bits and no more.  `symtable.py`
/// reads it in `is_global` and `is_local` alone, and in both only for a symbol
/// in the module block; `DEF_TYPE_PARAM` is set only on names inside a `type
/// parameters` block, so the wider mask never reached a reader.
const DEF_BOUND: SymbolFlags = SymbolFlags::from_bits_retain(
    SymbolFlags::DEF_LOCAL.bits() | SymbolFlags::DEF_PARAM.bits() | SymbolFlags::DEF_IMPORT.bits(),
);

fn table_type(table: &SymbolTable) -> i32 {
    match table.typ {
        CompilerScope::Function
        | CompilerScope::AsyncFunction
        | CompilerScope::Lambda
        | CompilerScope::Comprehension => TYPE_FUNCTION,
        CompilerScope::Class => TYPE_CLASS,
        CompilerScope::Module => TYPE_MODULE,
        CompilerScope::Annotation => TYPE_ANNOTATION,
        CompilerScope::TypeAlias => TYPE_TYPE_ALIAS,
        CompilerScope::TypeParams => TYPE_TYPE_PARAMETERS,
        CompilerScope::TypeVariable => TYPE_TYPE_VARIABLE,
    }
}

fn append_public_children(table: &SymbolTable, children_slot: usize) {
    for child in &table.sub_tables {
        if child.comp_inlined {
            // PEP 709 can nest inlined comprehensions. Flatten every inlined
            // layer and expose only the first non-inlined descendants.
            append_public_children(child, children_slot);
            continue;
        }
        let data = table_data(child);
        let data_roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(data);
        let data_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            pyre_object::listobject::w_list_append(
                pyre_object::gc_roots::shadow_stack_get(children_slot),
                pyre_object::gc_roots::shadow_stack_get(data_slot),
            );
        }
        drop(data_roots);
    }
}

/// Convert one compiler table into
/// `(name, type, lineno, nested, symbols, varnames, children)`.
fn table_data(table: &SymbolTable) -> PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();

    let symbols = pyre_object::dictmultiobject::w_dict_new();
    let _ = pyre_object::gc_roots::pin_root(symbols);
    let symbols_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    for (name, symbol) in &table.symbols {
        // CPython stores the resolved scope in the high bits of the public
        // symbol flags (`ste_symbols`); RustPython keeps it in a separate enum.
        let flags = i32::from((symbol.flags & PUBLIC_SYMBOL_FLAGS).bits())
            | (symbol.scope.as_i32() << SCOPE_OFF);
        // Root the newly allocated value before reloading `symbols`: the int
        // allocation may run a moving minor collection and update
        // `symbols_slot`, so evaluating it inline as the third call argument
        // would leave the first argument holding the pre-collection address.
        let value_roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(flags as i64));
        let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                pyre_object::gc_roots::shadow_stack_get(symbols_slot),
                name,
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
        drop(value_roots);
    }

    let varnames = pyre_object::listobject::w_list_new(Vec::new());
    let _ = pyre_object::gc_roots::pin_root(varnames);
    let varnames_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    for name in &table.varnames {
        let value_roots = pyre_object::gc_roots::push_roots();
        let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
        let value_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        unsafe {
            pyre_object::listobject::w_list_append(
                pyre_object::gc_roots::shadow_stack_get(varnames_slot),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
        drop(value_roots);
    }

    let children = pyre_object::listobject::w_list_new(Vec::new());
    let _ = pyre_object::gc_roots::pin_root(children);
    let children_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    append_public_children(table, children_slot);

    // symtable.py SymbolVisitor.visit_Lambda / _visit_comprehension name
    // anonymous scopes with angle brackets. RustPython keeps the compiler's
    // internal spelling here; expose the same public name as its _symtable
    // PySymbolTable.name adapter, without renaming ordinary user functions.
    let name = match table.typ {
        CompilerScope::Lambda => "<lambda>",
        CompilerScope::Comprehension if table.name.as_str() == "genexpr" => "<genexpr>",
        _ => table.name.as_str(),
    };
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_str_new_managed(name));
    let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(table_type(table) as i64));
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let _ = pyre_object::gc_roots::pin_root(pyre_object::w_int_new(table.line_number as i64));
    let line_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let mut fields = pyre_object::gc_roots::RootedItems::new();
    fields.push(pyre_object::gc_roots::shadow_stack_get(name_slot));
    fields.push(pyre_object::gc_roots::shadow_stack_get(type_slot));
    fields.push(pyre_object::gc_roots::shadow_stack_get(line_slot));
    fields.push(pyre_object::w_bool_from(table.is_nested));
    fields.push(pyre_object::gc_roots::shadow_stack_get(symbols_slot));
    fields.push(pyre_object::gc_roots::shadow_stack_get(varnames_slot));
    fields.push(pyre_object::gc_roots::shadow_stack_get(children_slot));
    pyre_object::tupleobject::w_tuple_new(fields.take())
}

fn symtable_data(args: &[PyObjectRef]) -> pyre_interpreter::PyResult {
    // `_symtable_symtable_impl`: three positional arguments plus kw-only
    // `module`. The Python wrapper always forwards the fourth slot.
    if args.len() != 3 && args.len() != 4 {
        return Err(pyre_interpreter::PyError::type_error(format!(
            "symtable() takes exactly 3 positional arguments ({} given)",
            args.len()
        )));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    for &arg in args {
        let _ = pyre_object::gc_roots::pin_root(arg);
    }
    // `_symtable_symtable_impl`: `modname == Py_None` becomes NULL so a
    // SyntaxWarning's filter still derives the module from the filename;
    // a str is `_PyErr_EmitSyntaxWarning`'s `module`.
    let compile_module = {
        let module = if args.len() == 4 {
            pyre_object::gc_roots::shadow_stack_get(base + 3)
        } else {
            pyre_object::PY_NULL
        };
        if module.is_null() || unsafe { pyre_object::is_none(module) } {
            pyre_object::PY_NULL
        } else if unsafe { pyre_object::is_str(module) } {
            module
        } else {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "symtable() argument 'module' must be str or None, not {}",
                pyre_interpreter::type_methods::arg_type_name(module)
            )));
        }
    };
    let _ = pyre_object::gc_roots::pin_root(compile_module);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_filename = unsafe {
        let arg = pyre_object::gc_roots::shadow_stack_get(base + 1);
        if pyre_object::is_str(arg) {
            arg
        } else if pyre_object::is_bytes(arg) {
            // The filename takes the filesystem decode, so a byte the encoding
            // cannot spell is reported here rather than renamed.
            let data = pyre_object::bytesobject::bytes_like_data(arg);
            pyre_object::w_str_from_wtf8_managed(pyre_interpreter::typedef::fsdecode_wtf8(data)?)
        } else {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "expected str, got {} object",
                pyre_interpreter::type_methods::arg_type_name(arg)
            )));
        }
    };
    let _ = pyre_object::gc_roots::pin_root(w_filename);
    let filename_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    // The compiler dependency still requires a Rust `str`; retain this only
    // as its internal source path and restore the surrogate-preserving Python
    // object on any resulting SyntaxError below.
    let filename = unsafe {
        pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(filename_slot))
            .to_string_lossy()
            .into_owned()
    };
    let source = unsafe {
        let arg = pyre_object::gc_roots::shadow_stack_get(base);
        if pyre_object::is_str(arg) {
            pyre_interpreter::baseobjspace::text_w(arg)?.to_owned()
        } else if pyre_object::is_bytes(arg) {
            pyre_interpreter::compile::decode_source_bytes(
                pyre_object::bytesobject::bytes_like_data(arg),
                pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(filename_slot)),
                false,
            )?
        } else {
            return Err(pyre_interpreter::PyError::type_error(format!(
                "expected str, got {} object",
                pyre_interpreter::type_methods::arg_type_name(arg)
            )));
        }
    };
    let mode_text =
        pyre_interpreter::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(base + 2))?;
    let mode = mode_text
        .parse::<rustpython_compiler::Mode>()
        .map_err(|err| pyre_interpreter::PyError::value_error(err.to_string()))?;
    // `_Py_SymtableStringObjectFlags` parses through
    // `_PyParser_ASTFromString`, which emits tokenizer and escape warnings
    // with the same `module` `compile()` uses.
    let filename_wtf8 = unsafe {
        pyre_object::w_str_get_wtf8(pyre_object::gc_roots::shadow_stack_get(filename_slot))
    }
    .to_owned();
    pyre_interpreter::syntax_warnings::emit_tokenizer_syntax_warnings_and_module(
        &source,
        &filename_wtf8,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    )?;
    pyre_interpreter::syntax_warnings::emit_escape_warnings_and_module(
        &source,
        &filename_wtf8,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    )?;
    let table =
        rustpython_compiler::compile_symtable(&source, mode, &filename).map_err(|compile_err| {
            let mut err =
                pyre_interpreter::builtins::compile_err_to_syntax_error(compile_err, &source, mode);
            err.replace_syntax_error_filename(pyre_object::gc_roots::shadow_stack_get(
                filename_slot,
            ));
            err
        })?;
    Ok(table_data(&table))
}

pyre_interpreter::py_module! {
    "_symtable",
    interpleveldefs: {
        "USE" => pyre_object::w_int_new(SymbolFlags::USE.bits() as i64),
        "DEF_GLOBAL" => pyre_object::w_int_new(SymbolFlags::DEF_GLOBAL.bits() as i64),
        "DEF_LOCAL" => pyre_object::w_int_new(SymbolFlags::DEF_LOCAL.bits() as i64),
        "DEF_PARAM" => pyre_object::w_int_new(SymbolFlags::DEF_PARAM.bits() as i64),
        "DEF_NONLOCAL" => pyre_object::w_int_new(SymbolFlags::DEF_NONLOCAL.bits() as i64),
        "DEF_FREE_CLASS" => pyre_object::w_int_new(SymbolFlags::DEF_FREE_CLASS.bits() as i64),
        "DEF_IMPORT" => pyre_object::w_int_new(SymbolFlags::DEF_IMPORT.bits() as i64),
        "DEF_ANNOT" => pyre_object::w_int_new(SymbolFlags::DEF_ANNOT.bits() as i64),
        "DEF_COMP_ITER" => pyre_object::w_int_new(SymbolFlags::DEF_COMP_ITER.bits() as i64),
        "DEF_TYPE_PARAM" => pyre_object::w_int_new(SymbolFlags::DEF_TYPE_PARAM.bits() as i64),
        "DEF_COMP_CELL" => pyre_object::w_int_new(SymbolFlags::DEF_COMP_CELL.bits() as i64),
        "DEF_BOUND" => pyre_object::w_int_new(DEF_BOUND.bits() as i64),
        "SCOPE_OFF" => pyre_object::w_int_new(SCOPE_OFF as i64),
        "SCOPE_MASK" => pyre_object::w_int_new(
            (SymbolFlags::DEF_GLOBAL.bits()
                | SymbolFlags::DEF_LOCAL.bits()
                | SymbolFlags::DEF_PARAM.bits()
                | SymbolFlags::DEF_NONLOCAL.bits()) as i64
        ),
        "LOCAL" => pyre_object::w_int_new(SymbolScope::Local.as_i32() as i64),
        "GLOBAL_EXPLICIT" =>
            pyre_object::w_int_new(SymbolScope::GlobalExplicit.as_i32() as i64),
        "GLOBAL_IMPLICIT" =>
            pyre_object::w_int_new(SymbolScope::GlobalImplicit.as_i32() as i64),
        "FREE" => pyre_object::w_int_new(SymbolScope::Free.as_i32() as i64),
        "CELL" => pyre_object::w_int_new(SymbolScope::Cell.as_i32() as i64),
        "TYPE_FUNCTION" => pyre_object::w_int_new(TYPE_FUNCTION as i64),
        "TYPE_CLASS" => pyre_object::w_int_new(TYPE_CLASS as i64),
        "TYPE_MODULE" => pyre_object::w_int_new(TYPE_MODULE as i64),
        "TYPE_ANNOTATION" => pyre_object::w_int_new(TYPE_ANNOTATION as i64),
        "TYPE_TYPE_ALIAS" => pyre_object::w_int_new(TYPE_TYPE_ALIAS as i64),
        "TYPE_TYPE_PARAMETERS" => pyre_object::w_int_new(TYPE_TYPE_PARAMETERS as i64),
        "TYPE_TYPE_VARIABLE" => pyre_object::w_int_new(TYPE_TYPE_VARIABLE as i64),
    },
    inline_app: {
        r#"
class _Entry:
    __slots__ = (
        "name", "type", "lineno", "nested", "symbols", "varnames", "children",
        "id", "__weakref__",
    )

    def __init__(self, data):
        (self.name, self.type, self.lineno, self.nested,
         self.symbols, self.varnames, children) = data
        self.children = [_entry(child) for child in children]
        self.id = id(self)

    def __repr__(self):
        return (
            f"<symtable entry {self.name}({self.id}), "
            f"line {self.lineno}>"
        )


def _entry(data):
    return _Entry(data)


def symtable(source, filename, startstr, /, *, module=None):
    import _symtable
    return _entry(_symtable._symtable_data(source, filename, startstr, module))
"# => ["_Entry", "_entry", "symtable"],
    },
    functions: {
        "_symtable_data" / 4 = symtable_data,
    },
}
