#!/usr/bin/env python3
"""The C declarations pyre's extension ABI publishes, and where they come from.

Three commands over one shared model:

    snapshot <cpython-Include-dir>   rewrite the recorded CPython declarations
    check                            every export against that record
    generate                         write `pyre_decl.h` from both

`check` is the gate. An extension is compiled against CPython's own headers on
every other implementation, so a parameter pyre declares differently is not a
style difference -- it is a calling-convention mismatch that no test in this
repo would otherwise see. The failure is silent and platform-dependent: a
`long` third argument passed where the callee reads a `Py_ssize_t` agrees on
LP64 and disagrees on Windows.

The record is a file rather than a path into a CPython checkout because CI has
no CPython source. Regenerating it is how the target version is bumped, and the
diff is then the ABI change that bump makes.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
CPYEXT = ROOT / "pyre/pyre-interpreter/src/cpyext"
HEADER_DIR = ROOT / "include/pyre3.15t"
RECORD = pathlib.Path(__file__).resolve().parent / "cpython-abi.txt"
GENERATED = HEADER_DIR / "pyre_decl.h"


# ── reading C ──────────────────────────────────────────────────────────────

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    # A `printf`-checking attribute sits between the parameter list and the
    # semicolon, so a declaration that carries one reads as a parameter list
    # ending at the attribute's own closing paren.  It says nothing about the
    # calling convention, so it goes before anything is matched.
    return re.sub(r"Py_GCC_ATTRIBUTE\s*\(\((?:[^()]|\([^()]*\))*\)\)", " ", text,
                  flags=re.S)


def tidy(text):
    """One spelling for one type: `PyObject**` and `PyObject * *` both fold."""
    text = re.sub(r"\bregister\b", " ", text)
    text = re.sub(r"\s*\*\s*", " * ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Re-join the stars a declarator wrote apart, keeping `T *const *` readable.
    text = re.sub(r"\* (?=\*)", "*", text)
    return re.sub(r"\* const", "*const", text).strip()


def split_commas(text):
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
            continue
        depth += ch in "(["
        depth -= ch in ")]"
        cur += ch
    if cur.strip():
        out.append(cur)
    return out


NAMED = re.compile(r"^(.*?[\w\]\)\*])\s+([A-Za-z_]\w*)$")
KEYWORDS = {"void", "int", "char", "long", "short", "unsigned", "signed",
            "float", "double", "const"}


def param_type(text):
    """A parameter's type with its name dropped, since a declaration needs none."""
    text = tidy(text)
    if text in ("", "void"):
        return "void"
    if "(" in text:  # a function pointer spelled inline
        # Only the declarator -- `(*name)` -- holds a name to drop.  An
        # unanchored match would take the first identifier before any `)`,
        # which is a type in `void (*)(int, int)` and a trailing attribute
        # clause in `va_list) Py_GCC_ATTRIBUTE((format(printf, 1, 0))`.
        return re.sub(r"\(\s*(\*+)\s*[A-Za-z_]\w*\s*\)", r"(\1)", text, count=1)
    text = re.sub(r"\[\s*\]", " *", text)
    named = NAMED.match(text)
    if named and named.group(2) not in KEYWORDS:
        return tidy(named.group(1))
    return text


DECLARATION = re.compile(
    r"PyAPI_FUNC\s*\(\s*(?P<ret>[^;]*?)\s*\)\s*(?P<stars>\**)\s*"
    r"(?:_Py_NO_RETURN\s*)?(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*?)\)\s*;",
    re.S)


DATA = re.compile(
    r"PyAPI_DATA\s*\(\s*(?P<type>[^;()]*?)\s*\)\s*(?P<stars>\**)\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*;",
    re.S)


def read_data(paths):
    """name -> the C type of every `PyAPI_DATA` object the headers declare."""
    found = {}
    for path in paths:
        text = strip_comments(path.read_text(errors="replace"))
        for match in DATA.finditer(text):
            found.setdefault(match.group("name"),
                             tidy(match.group("type") + " " + match.group("stars")))
    return found


def read_declarations(paths):
    found = {}
    for path in paths:
        text = strip_comments(path.read_text(errors="replace"))
        for match in DECLARATION.finditer(text):
            params = [param_type(p) for p in split_commas(match.group("params"))]
            found.setdefault(match.group("name"),
                             (params or ["void"], tidy(match.group("ret") + match.group("stars"))))
    return found


# ── reading Rust ───────────────────────────────────────────────────────────

EXPORT = re.compile(
    r'#\[unsafe\(no_mangle\)\]\s*pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+'
    r"(?P<name>\w+)\s*\((?P<params>.*?)\)\s*(?:->\s*(?P<ret>[^{]+?))?\s*\{",
    re.S)

SCALARS = {
    "c_char": "char", "c_int": "int", "c_uint": "unsigned int",
    "c_long": "long", "c_ulong": "unsigned long",
    "c_longlong": "long long", "c_ulonglong": "unsigned long long",
    "c_double": "double", "c_float": "float",
    "isize": "Py_ssize_t", "usize": "size_t",
    "u8": "unsigned char", "i8": "signed char",
    "i32": "int32_t", "u32": "uint32_t", "i64": "int64_t", "u64": "uint64_t",
    # Named aliases the Rust side spells the same way the header does, so the
    # comparison is against the reference name rather than its width.
    "Py_UCS4": "Py_UCS4", "wchar_t": "wchar_t", "PyTime_t": "PyTime_t",
    # A Rust function that never returns has no C spelling of its own: the
    # header declares it `void` and marks the fact separately.
    "!": "void",
}

STRUCTS = {
    "CPyObject": "PyObject", "CPyVarObject": "PyVarObject",
    "CPyTypeObject": "PyTypeObject", "CPyBuffer": "Py_buffer",
    "CPyModuleDef": "PyModuleDef", "CPyTypeSpec": "PyType_Spec",
    "CPyMethodDef": "PyMethodDef", "CPyMemberDef": "PyMemberDef",
    "CPyGetSetDef": "PyGetSetDef", "c_void": "void",
    "CPyThreadState": "PyThreadState", "CPyComplex": "Py_complex",
    "CPyFrameObject": "PyFrameObject",
    "CPyInterpreterState": "PyInterpreterState",
    "CPyMutex": "PyMutex",
    "CPyCriticalSection": "PyCriticalSection",
    "CPyUnicodeWriter": "PyUnicodeWriter",
    "CPyDateTimeCAPI": "PyDateTime_CAPI",
    "CPySlot": "PySlot",
}


class Unmapped(Exception):
    """A Rust type with no recorded C spelling, which must not be guessed."""


def rust_to_c(rust):
    """`*mut CPyObject` is `PyObject *`; `*const *mut T` is `T *const *`."""
    rust = " ".join(rust.split())
    if rust in ("", "()"):
        return "void"
    for prefix, qualified in (("*mut ", False), ("*const ", True)):
        if rust.startswith(prefix):
            inner = rust_to_c(rust[len(prefix):])
            if not qualified:
                return inner + ("*" if inner.endswith("*") else " *")
            return inner + "const *" if inner.endswith("*") else "const " + inner + " *"
    bare = rust.split("::")[-1]
    for table in (SCALARS, STRUCTS):
        if bare in table:
            return table[bare]
    raise Unmapped(rust)


def read_exports():
    """(module, name, [C param types], C return type) for every cpyext export."""
    for path in sorted(CPYEXT.glob("*.rs")):
        for match in EXPORT.finditer(path.read_text()):
            params = []
            for piece in split_commas(match.group("params")):
                if piece.strip():
                    params.append(rust_to_c(piece.split(":", 1)[1]))
            ret = match.group("ret")
            yield (path.stem, match.group("name"), params or ["void"],
                   rust_to_c(ret) if ret else "void")


# ── comparing the two ──────────────────────────────────────────────────────

FUNCTION_POINTER = re.compile(r"typedef\s+[^;]*?\(\s*\*\s*([A-Za-z_]\w*)\s*\)\s*\(", re.S)
ENUM = re.compile(r"typedef\s+enum\b[^;{]*\{.*?\}\s*([A-Za-z_]\w*)\s*;", re.S)
ALIAS = re.compile(r"typedef\s+([A-Za-z_][\w \t]*?)\s+([A-Za-z_]\w*)\s*;")
# `typedef void *PyThread_type_lock;` -- the alias names a pointer, which the
# word-only ALIAS above cannot see because of the star.
POINTER_ALIAS = re.compile(
    r"typedef\s+(?:(?:struct|union|enum)\s+)?[A-Za-z_][\w \t]*?\s*\*\s*([A-Za-z_]\w*)\s*;")
# `#define PY_TIMEOUT_T long long` -- an object-like macro whose body is only
# type words stands for a type the same way a typedef does.
TYPE_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+"
                        r"((?:unsigned|signed|const|long|short|int|char|float|double|void"
                        r"|[A-Za-z_]\w*)(?:[ \t]+(?:unsigned|signed|long|short|int|char"
                        r"|float|double|void|[A-Za-z_]\w*))*[ \t]*\**)[ \t]*$", re.M)


INLINE = re.compile(
    r"^static\s+inline\s+(?P<ret>[A-Za-z_][\w\s]*?)\s*(?P<stars>\**)\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{]*?)\)\s*\{",
    re.M | re.S)


def read_header_inlines():
    """The entry points implemented in the header rather than exported.

    `PyArg_ParseTuple`, `Py_BuildValue` and the other variadics are C in
    `Python.h`, because pyre ships no companion library and rustc's
    `c_variadic` is unstable. They are entry points an extension calls by the
    same name and the same convention as any export, so they are checked
    against the record the same way -- otherwise the gate reports "every export
    matches" while the half of the ABI it cannot see drifts.
    """
    for path in sorted(HEADER_DIR.glob("*.h")):
        text = strip_comments(path.read_text(errors="replace"))
        for match in INLINE.finditer(text):
            params = [param_type(p) for p in split_commas(match.group("params"))]
            ret = tidy(match.group("ret") + " " + match.group("stars"))
            yield path.name, match.group("name"), params or ["void"], ret


C_SOURCE_DIR = CPYEXT / "src"

# `stars` takes a `const` between two of them, so that a definition answering
# `PyObject *const *` is read rather than passed over as no definition at all.
C_DEFINITION = re.compile(
    r"^(?P<ret>[A-Za-z_][A-Za-z0-9_ ]*?)\s*(?P<stars>\**(?:\s*const\s*\**)?)\s*"
    r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{]*?)\)\s*\{",
    re.M | re.S)


def read_c_definitions():
    """The entry points whose bodies are C rather than Rust.

    A variadic entry point cannot have a Rust body -- no Rust compiler walks a
    `va_list` -- so `PyArg_ParseTuple`, `Py_BuildValue` and their peers are C
    translation units compiled into the interpreter and exported from it. They
    are entry points an extension resolves like any other, so they are checked
    against the record the same way; the headers carry their declarations and
    this is where the definitions are.
    """
    for path in sorted(C_SOURCE_DIR.glob("*.c")):
        text = strip_comments(path.read_text(errors="replace"))
        for match in C_DEFINITION.finditer(text):
            if match.group("ret").split()[0] == "static":
                continue
            params = [param_type(p) for p in split_commas(match.group("params"))]
            ret = tidy(match.group("ret") + " " + match.group("stars"))
            yield path.name, match.group("name"), params or ["void"], ret


HAND_DECLARED = re.compile(r"^PyAPI_FUNC\([^)]*\)\s*(?P<name>[A-Za-z_]\w*)\s*\(", re.M)
RENAME = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(?P<name>[A-Za-z_]\w*)[ \t(]", re.M)


def read_renamed_exports():
    """The exports a hand-written header declares and then renames.

    `lock.h` declares `PyMutex_Lock` and follows the inline fast path with
    `#define PyMutex_Lock _PyMutex_Lock`, so that a caller reaches the export
    only on the contended path.  The declaration has to come before that
    rename, and one after it would name the inline function instead -- a second
    declaration of a `static inline` already defined, carrying an attribute it
    did not have.  So a renamed export is declared where it is renamed, and
    left out of the generated declarations.
    """
    for path in sorted(HEADER_DIR.glob("*.h")):
        if path.name == GENERATED.name:
            continue
        text = strip_comments(path.read_text(errors="replace"))
        declared = {match.group("name") for match in HAND_DECLARED.finditer(text)}
        for match in RENAME.finditer(text):
            if match.group("name") in declared:
                yield match.group("name")


INTERNAL_HEADER_DIR = HEADER_DIR / "internal"


def read_internal_declarations():
    """The entry points only a header under `internal/` declares.

    CPython keeps these out of an installed tree, and pyre ships them on the
    same terms.  A declaration in the generated header would undo that: it is
    reached by including `Python.h` alone, which is exactly the extension the
    internal header is meant to be closed to.
    """
    for path in sorted(INTERNAL_HEADER_DIR.glob("*.h")):
        text = strip_comments(path.read_text(errors="replace"))
        for match in HAND_DECLARED.finditer(text):
            yield match.group("name")


STATIC = re.compile(
    r"#\[unsafe\(no_mangle\)\]\s*pub\s+static\s+(?:mut\s+)?"
    r"(?P<name>\w+)\s*:\s*(?P<type>[^=]+?)\s*=",
    re.S)

MACRO = re.compile(
    r"macro_rules!\s+(?P<macro>\w+)\s*\{.*?"
    r"pub\s+static\s+(?:mut\s+)?\$(?P<variable>\w+)\s*:\s*(?P<type>[^=]+?)\s*=",
    re.S)


def read_statics():
    """name -> the C type of every global the cpyext layer defines.

    A global a table-driven macro declares carries its attribute inside the
    expansion, so the plain scan cannot see it; the macro's own body says what
    type it declares and its invocation says under which names.
    """
    found = {}
    for path in sorted(CPYEXT.glob("*.rs")):
        text = path.read_text()
        for match in STATIC.finditer(text):
            found[match.group("name")] = rust_to_c(match.group("type"))
        for macro in MACRO.finditer(text):
            c_type = rust_to_c(macro.group("type"))
            for name in macro_table_names(text, macro.group("macro")):
                found[name] = c_type
    return found


def macro_table_names(text, macro):
    """The left-hand side of every row the named macro is invoked with."""
    body = text.split(f"{macro}! {{")
    if len(body) < 2:
        return []
    names = []
    for line in body[1].split("\n}")[0].splitlines():
        name = line.strip().split(" =>")[0]
        if name and all(c.isalnum() or c == "_" for c in name):
            names.append(name)
    return names


def read_typedefs(paths):
    """name -> what it stands for, so an alias is not read as a distinct type."""
    table = {}
    for path in paths:
        text = strip_comments(path.read_text(errors="replace"))
        for match in FUNCTION_POINTER.finditer(text):
            table[match.group(1)] = "void *"
        for match in ENUM.finditer(text):
            table[match.group(1)] = "int"
        for match in ALIAS.finditer(text):
            base = " ".join(match.group(1).split())
            if not base.startswith(("struct", "enum", "union")):
                table.setdefault(match.group(2), base)
        for match in POINTER_ALIAS.finditer(text):
            table.setdefault(match.group(1), "void *")
        for match in TYPE_MACRO.finditer(text):
            name, base = match.group(1), " ".join(match.group(2).split())
            if name != base:
                table.setdefault(name, base)
    return table


# Every pointer occupies one slot and is passed the same way, so the pointee is
# not part of the calling convention. Integer widths are.
POINTER_SLOT = "a pointer"
WIDTH_ALIASES = {"ssize_t": "Py_ssize_t", "intptr_t": "Py_ssize_t"}


def abi_slot(c_type, typedefs):
    text = tidy(c_type)
    if text.endswith("*") or text.endswith("const"):
        return POINTER_SLOT
    text = text.replace("const ", "").strip()
    for _ in range(8):
        if text not in typedefs or typedefs[text] == text:
            break
        text = typedefs[text]
        if text.endswith("*"):
            return POINTER_SLOT
    return WIDTH_ALIASES.get(text, text)


def load_record():
    if not RECORD.exists():
        sys.exit(f"{RECORD} is missing; run `{sys.argv[0]} snapshot <Include dir>` first")
    declarations, data, typedefs, section = {}, {}, {}, None
    for line in RECORD.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            if line.startswith("# ["):
                section = line[3:].rstrip("]\n")
            continue
        name, _, rest = line.partition(" :: ")
        if section == "typedefs":
            typedefs[name] = rest
            continue
        if section == "data":
            data[name] = rest
            continue
        args, _, ret = rest.partition(" -> ")
        # The record was written with `split_commas`, so a parameter that
        # carries a comma of its own -- a function pointer's own list -- must
        # be read back with the same depth-aware split.
        declarations[name] = ([a.strip() for a in split_commas(args)], ret.strip())
    return declarations, data, typedefs


# ── commands ───────────────────────────────────────────────────────────────

def command_snapshot(args):
    include = pathlib.Path(args.include)
    headers = sorted(include.glob("*.h")) + sorted(include.glob("cpython/*.h"))
    if not headers:
        sys.exit(f"no headers under {include}")
    declarations = read_declarations(headers)
    data = read_data(headers)
    typedefs = read_typedefs(sorted(include.rglob("*.h")))
    lines = [
        "# The declarations pyre's extension ABI is measured against.",
        "# Written by scripts/cpyext-abi.py snapshot; do not edit by hand.",
        f"# source: CPython {args.version} Include/*.h and Include/cpython/*.h",
        "",
        "# [declarations]",
    ]
    lines += [f"{n} :: {', '.join(a)} -> {r}" for n, (a, r) in sorted(declarations.items())]
    lines += ["", "# [data]"]
    lines += [f"{n} :: {c}" for n, c in sorted(data.items())]
    lines += ["", "# [typedefs]"]
    lines += [f"{n} :: {t}" for n, t in sorted(typedefs.items())]
    RECORD.write_text("\n".join(lines) + "\n")
    print(f"{len(declarations)} declarations, {len(data)} data objects, "
          f"{len(typedefs)} typedefs -> {RECORD}")
    return 0


def command_check(args):
    declarations, data, typedefs = load_record()
    # An entry point CPython does not declare is ordinary -- the macros it
    # spells over struct fields have to be calls here.  One that differs from a
    # declared name only in case is not: it is a misspelling, and an extension
    # calling the real name finds no symbol.
    by_lowercase = {n.lower(): n for n in declarations}
    disagree, converted, misspelled = [], [], []
    checked = {"export": 0, "header inline": 0, "C body": 0}
    entry_points = [("export", f"cpyext/{m}.rs", n, p, r)
                    for m, n, p, r in read_exports()]
    entry_points += [("header inline", f"{HEADER_DIR.name}/{h}", n, p, r)
                     for h, n, p, r in read_header_inlines()]
    entry_points += [("C body", f"cpyext/src/{f}", n, p, r)
                     for f, n, p, r in read_c_definitions()]
    for kind, where_defined, name, params, ret in entry_points:
        if name not in declarations:
            spelled = by_lowercase.get(name.lower())
            if spelled is not None:
                misspelled.append((where_defined, name, spelled))
            converted.append((where_defined, name))
            continue
        theirs, their_ret = declarations[name]
        checked[kind] += 1
        ours = [abi_slot(p, typedefs) for p in params]
        wanted = [abi_slot(p, typedefs) for p in theirs]
        if ours != wanted:
            disagree.append((where_defined, name, params, theirs, "arguments"))
        elif abi_slot(ret, typedefs) != abi_slot(their_ret, typedefs):
            disagree.append((where_defined, name, [ret], [their_ret], "return"))
    disagree += check_data(data, typedefs, misspelled)
    print(f"{checked['export']} exports, {checked['header inline']} header "
          f"inlines and {checked['C body']} C bodies checked against the "
          f"recorded CPython ABI; "
          f"{len(converted)} have no CPython declaration")
    for where_defined, name, spelled in misspelled:
        print(f"\n{name} is spelled {spelled} by CPython  [{where_defined}]")
    for where_defined, name, ours, theirs, where in disagree:
        print(f"\n{name} disagrees on its {where}  [{where_defined}]")
        print(f"    pyre    ({', '.join(ours)})")
        print(f"    cpython ({', '.join(theirs)})")
    if disagree or misspelled:
        print(f"\n{len(disagree) + len(misspelled)} entry point(s) do not match "
              f"the C declaration an extension is compiled against.")
        return 1
    print("every entry point matches.")
    return 0


# A mirror is a `PyObject` whatever it stands for, so a singleton CPython
# declares under the layout of its own value cannot agree: the pyre headers
# offer no `PyLongObject` for the declaration to name, and `Py_True` casts the
# address to `PyObject *` on both sides, so nothing an extension writes can see
# the difference.  `api.py:711-713` registers the same two as `PyObject*`.
DATA_DIVERGENCES = {"_Py_TrueStruct", "_Py_FalseStruct"}


def check_data(record, typedefs, misspelled):
    """Every `PyAPI_DATA` object the pyre headers declare, three ways.

    A data object is invisible to the entry-point pass above: nothing about it
    is a `PyAPI_FUNC`, and its Rust side is a `static` rather than a function.
    It is still ABI an extension resolves at `dlopen` time, and getting it
    wrong fails there rather than at build time, so it is checked here for the
    type CPython gives it and for a definition actually existing behind the
    declaration.
    """
    declared = read_data(sorted(HEADER_DIR.glob("*.h")))
    defined = read_statics()
    by_lowercase = {n.lower(): n for n in record}
    disagree = []
    for name, c_type in sorted(declared.items()):
        where_declared = f"{HEADER_DIR.name}/*.h"
        if name in DATA_DIVERGENCES:
            continue
        if name not in defined:
            disagree.append((where_declared, name, [c_type],
                             ["no cpyext static defines it"], "definition"))
            continue
        if abi_slot(defined[name], typedefs) != abi_slot(c_type, typedefs):
            disagree.append((where_declared, name, [c_type], [defined[name]], "type"))
            continue
        if name not in record:
            spelled = by_lowercase.get(name.lower())
            if spelled is not None:
                misspelled.append((where_declared, name, spelled))
            continue
        if abi_slot(c_type, typedefs) != abi_slot(record[name], typedefs):
            disagree.append((where_declared, name, [c_type], [record[name]], "type"))
    print(f"{len(declared) - len(DATA_DIVERGENCES & declared.keys())} data objects "
          f"checked against the recorded CPython ABI")
    return disagree


def command_generate(args):
    declarations, _, _ = load_record()
    withheld = set(read_renamed_exports()) | set(read_internal_declarations())
    by_module = {}
    for module, name, params, ret in read_exports():
        if name in withheld:
            continue
        # CPython's own spelling wherever it has one: `check` has already
        # established the two describe the same call, and the reference
        # spelling is the one an extension's own prototypes agree with.
        if name in declarations:
            theirs, their_ret = declarations[name]
            params, ret = theirs, their_ret
        by_module.setdefault(module, []).append((name, params, ret))

    out = [
        "/* The exported entry points, one block per `cpyext` module.",
        " *",
        " * Written by scripts/cpyext-abi.py generate; do not edit by hand.",
        " * A declaration here is CPython's own where CPython has one, so an",
        " * extension's prototypes and pyre's agree by construction.",
        " *",
        " * An export a hand-written header renames to an inline fast path is",
        " * left out: that header declares it ahead of the rename, which a",
        " * declaration here would come after.  So is one a header under",
        " * `internal/` declares, which is reached by including that header",
        " * and not by including `Python.h`.",
        " */",
        "#ifndef PYRE_DECL_H",
        "#define PYRE_DECL_H",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]
    for module in sorted(by_module):
        out.append(f"/* cpyext/{module}.rs */")
        for name, params, ret in sorted(by_module[module]):
            out.append(f"PyAPI_FUNC({ret}) {name}({', '.join(params)});")
        out.append("")
    out += ["#ifdef __cplusplus", "}", "#endif", "", "#endif /* !PYRE_DECL_H */"]

    text = "\n".join(out) + "\n"
    target = GENERATED
    if args.check:
        if not target.exists() or target.read_text() != text:
            print(f"{target} is not what `generate` produces; re-run without --check")
            return 1
        print(f"{target} is up to date")
        return 0
    target.write_text(text)
    print(f"{sum(len(v) for v in by_module.values())} declarations -> {target}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    snapshot = sub.add_parser("snapshot", help="rewrite the recorded CPython declarations")
    snapshot.add_argument("include", help="a CPython checkout's Include directory")
    snapshot.add_argument("--version", default="3.15", help="what to record as the source")
    snapshot.set_defaults(run=command_snapshot)

    check = sub.add_parser("check", help="every export against the record")
    check.set_defaults(run=command_check)

    generate = sub.add_parser("generate", help="write pyre_decl.h")
    generate.add_argument("--check", action="store_true",
                          help="fail if the checked-in file is not what this produces")
    generate.set_defaults(run=command_generate)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
