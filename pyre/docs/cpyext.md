# cpyext in pyre

Pyre follows PyPy's `pypy/module/cpyext` architecture.  It does not expose an
internal pyre object as a CPython `PyObject *`: the two layouts are unrelated.
Instead, a C extension sees a fixed-address, reference-counted mirror carrying
an `ob_pyre_link` to the interpreter's GC object.  The interpreter's global GC
root walker forwards that link while an external C reference owns it.

## Build option

Extension loading is behind the non-default `cpyext` cargo feature. The switch
itself is upstream's: `has_so_extension` (`pypy/module/imp/importing.py:60`)
reads `config.objspace.usemodules.cpyext` and gates `create_dynamic`
(`pypy/module/imp/interp_imp.py:49-51`). Build it with

```sh
cargo build --release -p pyrex --bin pyre-dynasm \
  --no-default-features --features dynasm,cpyext
```

The feature also decides whether `_imp.extension_suffixes()` answers with a
suffix. **That part is a deliberate divergence**: upstream
`extension_suffixes` (`pypy/module/imp/interp_imp.py:11-15`) answers
unconditionally, so a `--withoutmod-cpyext` PyPy still names a suffix whose
`create_dynamic` then raises. A suffix names a loader, so pyre keeps the two
together — without the feature the import path advertises nothing it cannot
load, which is also what pyre answered before extension loading existed.

The consequence is why the feature is off rather than on:
`importlib._bootstrap_external.EXTENSION_SUFFIXES` is that call's result, and
`test_importlib/extension` skips its whole suite while the list is empty. A
non-empty list un-skips 39 tests that load CPython's `_testsinglephase` and
`_testmultiphase`; the ABI header below builds neither yet. Turn the feature on
by default once slice 5 below lands and both modules build (slice 6 is Windows
packaging, which the os gates make irrelevant to the default).

## The header

`include/pyre3.15t/` is what an extension compiles against.
An extension includes `Python.h`; that file is only the include list, and the
content is split by topic the way `pypy/module/cpyext/include` and CPython's
own `Include` are -- `object.h`, `longobject.h`, `modsupport.h`, and so on, one
per `cpyext` module. `pytypedefs.h` names the shared types once so no
declaration waits on a definition, and `audit.h` carries the variadic
`PySys_Audit` the way CPython's own `Include/audit.h` does.

A header declares; it does not implement. The entry points that cannot have
Rust bodies -- the variadic ones, since rustc's `c_variadic` is unstable -- are
C translation units under `pyre-interpreter/src/cpyext/src/`, compiled into the
interpreter and exported from it, which is what `pypy/module/cpyext/src/` is.

`pyre_decl.h` holds the exported entry points and **is generated** by
`scripts/cpyext-abi.py` from the `#[unsafe(no_mangle)] extern "C"` functions
themselves, so a declaration cannot drift from its implementation. It is
included after the headers that name the types it uses and before the ones
defining `static inline` functions that call it. `refcount.h` is the first of
those: `Py_INCREF` expands to the `Py_IncRef` it declares, and `Py_NewRef` is an
inline function calling that macro.

The exports it leaves out are the three a header renamed. `lock.h` declares
`PyMutex_Lock` and then, past the inline fast path, `#define`s the name onto
it; a declaration here would come after that rename and so name the inline
function rather than the export.

The generator does not spell the declaration from the Rust signature where
CPython has one of its own. It emits **CPython's** declaration -- `PyLongObject
*`, `Py_hash_t`, `PyCapsule_Destructor` -- having first checked that the two
describe the same call. That check is `scripts/cpyext-abi.py check`, and it is
a CI gate:

```sh
python3 pyre/scripts/cpyext-abi.py check          # every entry point vs. CPython
python3 pyre/scripts/cpyext-abi.py generate --check
```

An *entry point* is either an export or a `static inline` function in the
header. Both reach the extension as a call it compiles against, so both are
checked; a header inline is the one an export cannot catch, because it is
written by hand rather than generated.

`check` also covers the exported **data** -- the `PyAPI_DATA` objects -- which
no part of the generated half can see: nothing about one is a `PyAPI_FUNC`, and
its Rust side is a `static` rather than a function. Each declared name is
checked three ways: that a `cpyext` static actually defines it, that the two
agree on its type, and that the type is the one CPython gives it. The first is
the one that matters, because a declaration with nothing behind it fails at
`dlopen` time in a loaded extension rather than at build time here. A static a
table-driven macro declares is read out of the macro's own body and its
invocation, the same way `pyrex/build.rs` reads them to export them.

`_Py_TrueStruct` and `_Py_FalseStruct` are the recorded exceptions: CPython
declares each as the `PyLongObject` its value is, and a mirror is a `PyObject`
whatever it stands for. `Py_True` casts the address on both sides, so nothing
an extension writes can see the difference.

It compares ABI slots rather than spellings, resolving CPython's typedefs, so
`Py_hash_t` and `Py_ssize_t` agree and `long` and `Py_ssize_t` do not. The
recorded declarations live in `scripts/cpython-abi.txt` because CI has no
CPython checkout; `snapshot` rewrites them from one, and that diff is the ABI
change a version bump makes.

This matters because nothing else here can see the failure. Every fixture in
this tree is compiled against pyre's own header, so a parameter declared at the
wrong width agrees with itself and only disagrees with the real world -- and
only on a platform where the widths differ. `PyModule_AddIntConstant` took a
`Py_ssize_t` where the reference declaration says `long`, which is the same on
LP64 and 32 bits against 64 on Windows.

The exports CPython does not declare are the macros it spells over struct
fields -- `PyLong_Check`, `PySequence_Fast_GET_ITEM`, `PyBytes_AS_STRING` --
which have to be calls here, plus the two `_PyPyre_*` helpers the header's own
`static inline` functions use.

`patchlevel.h` carries the version in parts -- `PY_MAJOR_VERSION`,
`PY_RELEASE_LEVEL`, `PY_RELEASE_SERIAL` -- and computes `PY_VERSION_HEX` from
them, so an extension testing any one of them reads the same number the runtime
reports. `PYRE_VERSION` is the interpreter's own version beside it, the slot
`PYPY_VERSION` fills upstream. `cpyext_methods` compares the whole set against
`sys.version_info`, `sys.hexversion` and `sys.pyre_version_info`.

`PYPY_VERSION` itself is what an extension branches on to ask whose object
layouts these headers describe, and it is defined -- except for a module cffi
generated, which `_cffi_include.h` marks with `_CFFI_` before it includes
`Python.h`. Such a module has two forms and picks between them on that macro:
one built out of `_cffi_exports` whose init symbol is `PyInit_`, and one
exporting `_cffi_pypyinit_` that hands its type context to a built-in backend.
The backend here is cffi's own `_cffi_backend` extension, which is the first
form's, so the macro is left out for that one translation unit and the module
compiles to it. `cpyext_cffi_mode` stands where such a module stands and
reports what it picked.

## What is implemented

The macOS/Linux slice is end-to-end: `extra_tests/cpyext/test_smoke_*.py`
imports a single-phase extension, `extra_tests/cpyext/test_methods.py` a
multi-phase one and `extra_tests/cpyext/test_types_*.py` one defining types,
all compiled from C (`pyrex/tests/fixtures/*.c`) against the header below by
`extra_tests/cpyext/run.py`. The dict-subclass, pystate, object-family, str,
exceptions, warnings, conversions, type-statics, runtime, small, locks and
writer scripts in that directory take their expectations from CPython 3.14.6
running the same script against the same fixture. Every fixture is compiled
with `-Werror`: it is written against these headers and nothing else, so a
warning in one is either the fixture calling an entry point wrongly or the
header declaring it wrongly.

- a pyre-specific Python 3.14 ABI header and extension suffix;
- `_imp.extension_suffixes()`, `_imp.create_dynamic()` and
  `_imp.exec_dynamic()`;
- `dlopen`/`dlsym` of `PyInit_<name>` with the library handle retained;
- the raw mirror layer (`cpyext/pyobject.rs`): the identity table, borrowed
  references owned by their container, the per-mirror byte cache behind
  `PyUnicode_AsUTF8` / `PyBytes_AsString`, and the immortal singletons
  `Py_None` / `Py_True` / `Py_False` / `Py_Ellipsis` / `Py_NotImplemented`;
- the type mirror `Py_TYPE(x)` answers with. A type an extension defined is its
  own `PyTypeObject` static, which `PyType_Ready` links in place and which is
  immortal because that storage is the library's; a type pyre defines gets a
  synthesized block carrying the ordinary link share, so a class the extension
  merely observed dies with the interpreter's own last reference to it. Its
  `tp_flags` carry `Py_TPFLAGS_HEAPTYPE` for a heap type, which is what decides
  whether a block holds a reference to its own type (`pyobject.py:91-93`,
  `object.py:72-73`) and whose storage the mirror is
  (`typeobject.py:716-722`);
- the 49 builtin `PyTypeObject` statics an extension names by address
  (`cpyext/typeobject.rs`): `PyList_Type`, `PyDict_Type`, `PyType_Type` and
  their siblings, the same family `api.py:746-790 build_exported_objects`
  registers. Each is storage whose address is fixed at link time and whose
  body the runtime fills in place, because C spells `&PyList_Type` -- a mirror
  allocated on demand can never be that. They are bound before the singletons:
  binding a singleton resolves its own `ob_type`, and a type reached before
  its static is entered would get a synthesized block instead, which is then
  the block `Py_TYPE(Py_True)` answers with forever. The address being the one
  the runtime hands out is what makes `Py_IS_TYPE(x, &PyList_Type)`,
  `PyObject_TypeCheck`, the `O!` argument format and `Py_tp_base` work;
- the two forms a mirror is handed out in before its interpreter object exists
  (`cpyext/unicodeobject.rs`, `cpyext/bytesobject.rs`): `PyUnicode_New(size,
  maxchar)` and `PyBytes_FromStringAndSize(NULL, size)` return an unlinked
  mirror over a buffer the caller fills, and `pyobject::realize` builds the
  `str` or `bytes` the first time it is read as a value, which is where
  `pyobject.py:330-337` reaches the type descriptor's `realize`. The type tests
  and the length are answered from the buffer, so asking them mid-fill does not
  decide the contents early;
- the C exception indicator (`cpyext/pyerrors.rs`): 69 `PyExc_*` class mirrors
  -- every one CPython declares outside `MS_WINDOWS`, `PyExc_EnvironmentError`
  and `PyExc_IOError` among them, both resolving to `OSError` --
  and `PyErr_SetString` / `SetObject` / `SetNone` / `Occurred` / `Clear` /
  `Fetch` / `Restore` / `ExceptionMatches` / `NoMemory` / `BadArgument` /
  `BadInternalCall`, plus the `PyErr_Format` half the header cannot do. The
  indicator is one normalized instance, so `PyErr_GetRaisedException` is that
  slot detached and `PyErr_SetRaisedException` is it replaced, both moving the
  reference rather than counting a new one, and the `Fetch` triple's three
  slots are all derived from it -- the class it is of, itself, and its
  `__traceback__`. `Restore` writes the traceback back onto the instance, which
  is what makes the pair lossless for a caller that saves the triple across
  work of its own. The three `Set*` spellings chain: an exception raised while
  another is being handled records the handled one as its `__context__`, as
  `_PyErr_SetObject` does, so the C caller's error does not hide it; `Restore`
  does not, the exception it is handed being an older one rather than a new
  raise;
- the exception a failed syscall becomes (`cpyext/pyerrors.rs`):
  `PyErr_SetFromErrno` and its three filename spellings, plus
  `PyErr_CheckSignals` over the signal state the interpreter already keeps.
  The class is *called* rather than consulted, as `errors.c:899` does, so
  `OSError` maps the code to its own subclass and a class outside the family
  is built from the same `(errno, strerror)` pair; a second filename is the
  fifth argument, the fourth being the Windows code that is 0 here. The
  `const char *` filename spelling reads the code before decoding the path,
  the decode being a call that can overwrite it;
- the audit events (`cpyext/sysmodule.rs`, `include/pyre3.15t/audit.h`):
  `PySys_AuditTuple` and the variadic `PySys_Audit` over it, which reach the
  same hooks `sys.addaudithook` installs. A hook is Python code, so a raise
  is what the `-1` reports;
- `PyImport_ImportModuleAttr` / `PyImport_ImportModuleAttrString`
  (`cpyext/import_.rs`), the import-then-`getattr` pair a C caller reaching
  into Python uses;
- the locks an extension holds while it works (`cpyext/lock.rs`,
  `include/pyre3.15t/lock.h`, `include/pyre3.15t/pythread.h`): the one-byte
  `PyMutex` the caller embeds, whose uncontended take and release are the
  header's own compare-exchange and whose contended halves are exported; and
  the allocated `PyThread_type_lock` with `PyThread_acquire_lock` /
  `acquire_lock_timed` / `release_lock` / `free_lock`, which is the binary
  semaphore one thread may hand to another rather than an owned mutex. Every
  wait that can block runs inside `gc_sync::before_external_block` -- the same
  region `Py_BEGIN_ALLOW_THREADS` puts a thread in -- so a thread asleep on a
  lock gives the GIL up and stops being one a collection waits for;
- `_PyBytes_Resize` (`cpyext/bytesobject.rs`), which is how a caller that took
  a buffer for an upper bound cuts it down to what it wrote. A mirror
  `PyBytes_FromStringAndSize(NULL, n)` handed out has no `bytes` behind it
  yet, so its buffer is resized where it lies and the same block is written
  back; a mirror that has one gets a new `bytes`, the object being immutable.
  Bytes past the old length are zero, where upstream leaves them holding
  whatever the allocator had if it can resize in place;
- `_PyErr_ChainExceptions1` (`cpyext/pyerrors.rs`), which makes what a caller
  just caught the context of whatever is already pending rather than letting
  one hide the other, and `Py_FatalError` over the exported
  `_Py_FatalErrorFunc`, a macro so that the report names the function the
  caller gave up in;
- the `str` an extension builds piece by piece (`cpyext/unicodewriter.rs`):
  `PyUnicodeWriter_Create` / `Discard` / `Finish` and the writes
  `WriteChar` / `WriteUTF8` / `WriteASCII` / `WriteWideChar` / `WriteUCS4` /
  `WriteStr` / `WriteRepr` / `WriteSubstring`, plus the variadic
  `PyUnicodeWriter_Format` written on the header over the `%`-format engine
  already there. The handle is opaque, so what it holds is this layer's own
  text rather than a partly built `str`: nothing is an object until `Finish`
  is asked for one, and a writer that is discarded made none. A refused write
  leaves the writer holding exactly what it held before it, so the caller may
  still finish or discard it;
- the container entry points that answer *whether* the key was there beside
  the value (`cpyext/dictobject.rs`, `cpyext/mapping.rs`,
  `cpyext/listobject.rs`, `cpyext/bytesobject.rs`): `PyDict_Pop` /
  `PyDict_PopString`, which take the key out and hand back a new reference,
  and answer the miss without hashing the key at all when the dictionary is
  empty; `PyMapping_GetOptionalItem` / `GetOptionalItemString`, which swallow
  the `KeyError` a lookup raised and let every other failure through;
  `PyList_Clear` and `PyList_Extend`, which reach the `list` type's own
  methods rather than a subclass override; `PyDictProxy_New`, the read-only
  view `types.MappingProxyType` builds, over anything with `__getitem__` that
  is not a list or a tuple; and `PyBytes_Join` / `PyBytes_Concat` /
  `PyBytes_ConcatAndDel`, the last two replacing the reference the caller
  handed over with the concatenation and leaving NULL there when it fails;
- the small entry points beside them (`cpyext/unicodeobject.rs`,
  `cpyext/dictobject.rs`, `cpyext/object.rs`, `cpyext/typeobject.rs`,
  `cpyext/genericaliasobject.rs`): the locale codec `PyUnicode_DecodeLocale` /
  `DecodeLocaleAndSize` / `EncodeLocale`, which takes only the `strict` and
  `surrogateescape` handlers and refuses the rest before it runs rather than
  reaching the codec registry, reporting what it cannot convert as the
  `locale` codec's own `decoding error` / `encoding error`;
  `PyUnicode_FromKindAndData`, whose units are code points of one of three
  widths, so a 2-byte unit in the surrogate range is one character rather than
  half an encoding; `Py_HashBuffer`, the hash `bytes` of the same content has;
  `PyDict_SetDefaultRef`, whose reference is a new one either way -- which is
  what separates it from the borrowing `PyDict_SetDefault` beside it -- and
  which leaves NULL there whenever the call fails; `Py_ReprEnter` /
  `Py_ReprLeave` over the interpreter's own mid-repr set, so a container
  reached from a C `tp_repr` and one reached from Python see the same
  recursion; `Py_GenericAlias`, what `origin[args]` builds; and
  `_PyType_Name`, the tail of `tp_name` after the last dot, which points into
  `tp_name` itself and so is freed by nobody;
- the *handled* exception beside it: `PyErr_GetHandledException` /
  `SetHandledException` and the triple spelling `PyErr_GetExcInfo` /
  `SetExcInfo`. The read walks the suspended generators' saved slots, so it
  answers the topmost handler's exception, while the write reaches the current
  execution context's own -- inside a generator those are different slots, as
  they are upstream. Only the value is ever stored: a later `GetExcInfo`
  derives the class and the traceback from it again, so whatever was passed as
  either is released and nothing more. `SetHandledException` borrows where
  every other setter in the family steals, and the empty state is the
  asymmetric one -- the class and traceback slots receive `None` while the
  value slot receives NULL, which is what tells it apart from `sys.exc_info()`;
- `PyErr_SetImportError` and `PyErr_SetImportErrorSubclass`, which build the
  instance by calling the class as `class(message, name=..., path=...,
  name_from=...)`. A subclass whose `__init__` does not take those three
  keywords therefore refuses, and one whose constructor raises leaves its own
  exception standing instead. Both always answer NULL, which is the convention
  `return PyErr_SetImportError(...)` relies on;
- the exception instance's own slots (`cpyext/exception.rs`):
  `PyException_GetTraceback` / `SetTraceback` / `GetCause` / `SetCause` /
  `GetContext` / `SetContext` / `GetArgs` / `SetArgs`, and the classification
  spellings `PyExceptionClass_Check` / `PyExceptionInstance_Check` /
  `PyExceptionClass_Name`. `SetCause` and `SetContext` write the typed slot
  directly rather than through `setattr`, because the attribute setters refuse
  anything that is not `None` or a `BaseException` instance and these entry
  points check nothing; `SetTraceback` does go through `setattr`, that slot
  being type-checked in C too. `PyExceptionInstance_Class` stays the macro it
  is upstream;
- the warnings entry points (`cpyext/warnings.rs`): `PyErr_WarnEx`,
  `PyErr_WarnExplicit` and `PyErr_WarnExplicitObject`, with the variadic
  `PyErr_WarnFormat`, `PyErr_ResourceWarning` and `PyErr_WarnExplicitFormat`
  written in the header over the two exported cores `_PyPyre_WarnUnicode` and
  `_PyPyre_WarnExplicitMessage`, for the reason the argument parsers are there
  too: rustc's `c_variadic` is unstable. They hand their arguments to the same `do_warn` /
  `do_warn_explicit` the `_warnings` module runs, so the filters, the
  `__warningregistry__` deduplication and `warnings.catch_warnings(record=True)`
  all see a warning an extension issued. Two consequences are visible from C: a
  NULL category is `RuntimeWarning`, and the category is never checked -- it is
  called with the message, so a class that is not a `Warning` is emitted under
  that class and one that is not callable refuses;
- the conversions at the C boundary (`cpyext/unicodeobject.rs`,
  `cpyext/osmodule.rs`): the named codecs `PyUnicode_Decode` /
  `DecodeASCII` / `DecodeLatin1` / `DecodeUTF8` and `PyUnicode_AsEncodedString`
  / `AsASCIIString` / `AsLatin1String` / `AsUTF8String`, each running the body
  `bytes.decode` and `str.encode` run so the codec set and the error handlers
  are the interpreter's own, and reaching it without the method lookup so that
  a `str` subclass defining `encode` does not change what these encode; the
  `wchar_t` forms `PyUnicode_FromWideChar` / `AsWideChar` /
  `AsWideCharString`, whose unit is one code point where a `wchar_t` is four
  bytes and one UTF-16 unit where it is two; and the filesystem encoding
  `PyUnicode_DecodeFSDefault` / `DecodeFSDefaultAndSize` / `EncodeFSDefault`
  with the two `O&` converters `PyUnicode_FSConverter` / `FSDecoder` over
  `PyOS_FSPath`. `PyUnicode_AsWideCharString` allocates through `PyMem_Malloc`,
  which is the allocator its caller frees it with;
- `PyIndex_Check`, and `Py_GetConstant` / `Py_GetConstantBorrowed` over one
  immortal mirror per constant, since the borrowed spelling hands out a mirror
  it took no reference to and two asks for the same identifier have to answer
  the same pointer;
- a parenthesised unit in the argument parser (`cpyext/src/getargs.c`)
  and the compat-mode `PyArg_Parse` over it, where the argument is the value
  itself rather than a tuple holding it and a format naming more than one unit
  is a `SystemError`. `PyOS_snprintf` and `PyOS_vsnprintf` are beside
  `PyErr_Format` in the C sources, being variadic for the same reason. The `O!`
  unit tests the layout -- `PyType_IsSubtype(Py_TYPE(arg), type)`, the way
  `convertsimple` does -- rather than asking `isinstance`, since the caller
  goes on to read the object through the fields that type declares and an
  object whose `__class__` merely answers with it has none of them;
- the `PyCFunction` carrier (`cpyext/methodobject.rs`) and the call bridge for
  `METH_NOARGS`, `METH_O`, `METH_VARARGS`, `METH_VARARGS | METH_KEYWORDS`,
  `METH_FASTCALL` and `METH_FASTCALL | METH_KEYWORDS`;
- the call entry points in the other direction (`cpyext/object.rs`):
  `PyObject_Call` / `CallObject` / `CallNoArgs` / `CallOneArg`, and the
  vectorcall spellings `PyObject_Vectorcall`, `PyObject_VectorcallMethod` and
  `PyVectorcall_Call`, which unpack the argument vector and go through the
  interpreter's own call path. Both halves of PEP 590 are taken: a type
  declaring `Py_TPFLAGS_HAVE_VECTORCALL` has the function read out of the
  instance at `tp_vectorcall_offset` and called with the values as they lie,
  ahead of the `tp_call` such a type is also required to carry
  (`_PyVectorcall_FunctionInline`), and the flag travels to a subclass that
  leaves `tp_call` alone. The
  variadic ones -- `PyObject_CallFunction` / `CallMethod`, the two `ObjArgs`
  spellings, `CallMethodNoArgs` / `CallMethodOneArg` and `PyVectorcall_NARGS`
  -- are `static inline` in the header, like `PyArg_ParseTuple`;
- single-phase `PyModule_Create2`, PEP 489 `PyModuleDef_Init` with
  `Py_mod_create` / `Py_mod_exec`, module state (`m_size`) and the
  `PyModule_Add*` family;
- the object protocol (`cpyext/object.rs`) and the primitive constructors and
  accessors for `int`, `bool`, `float`, `str`, `bytes`, `tuple`, `list` and
  `dict`;
- `bytearray` (`cpyext/bytearrayobject.rs`), `complex`
  (`cpyext/complexobject.rs`) and `weakref` (`cpyext/weakrefobject.rs`).
  `PyByteArray_AsString` answers with the payload itself rather than a copy, so
  a write through it reaches the object, with a terminating NUL in the backing
  allocation's spare capacity; the pointer stops being current as soon as the
  object is resized, as upstream's own comment states. `PyComplex_AsCComplex`
  and `PyComplex_FromCComplex` pass `Py_complex` by value, where upstream
  reaches it through a pointer because lltype cannot return a struct.
  `PyWeakref_GetObject` borrows the referent's mirror rather than taking the
  reference a container owns on an item's behalf -- that one lives until the
  container's mirror dies, which is the object a weak reference must not keep
  alive;
- the `str` operations an extension builds text with
  (`cpyext/unicodeobject.rs`): `Concat` / `Append` / `AppendAndDel`,
  `Substring`, `Join`, `FromOrdinal`, `FromObject`, `DecodeUTF8`,
  `InternFromString` / `InternInPlace`, `FindChar`, `Contains`, and the five
  comparisons `Compare` / `CompareWithASCIIString` / `RichCompare` / `Equal` /
  `EqualToUTF8`. `PyUnicode_DecodeUTF8` runs the body `bytes.decode` runs, so
  every error handler the interpreter has is reachable rather than the three
  a hand-written decoder would name;
- the `%`-format engine and `PyUnicode_FromFormat` / `FromFormatV` over it
  (`cpyext/src/unicodeobject.c`), which `PyErr_Format` is now written on. The
  conversions
  are assembled as `str` objects rather than as bytes, because `%6S` pads to a
  count of characters and only the interpreter knows how many a conversion
  produced. That is also what makes `%c` a code point, `%A` `ascii()` rather
  than `repr()`, `%V` the one conversion taking two arguments, `%T` / `%N` a
  type's qualified name, and a conversion this does not describe a
  `SystemError` rather than something handed to `snprintf`;
- the GIL an extension hands back and forth (`cpyext/pystate.rs`):
  `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` and the
  `Py_BLOCK_THREADS` pair, `PyEval_SaveThread` / `PyEval_RestoreThread` /
  `PyEval_AcquireThread` / `PyEval_ReleaseThread`, `PyGILState_Ensure` /
  `PyGILState_Release` / `PyGILState_Check` / `PyGILState_GetThisThreadState`,
  and `PyThreadState_Get` / `PyThreadState_Swap`. These are the runtime's own
  boundary guards: a thread holds the GIL for as long as it runs pyre code, so
  giving it up is `before_external_block` and taking it is
  `enter_external_callback_from_foreign_thread`, which registers a thread the
  extension owns. `PyThreadState` is opaque, one per thread.
  `PyGILState_Ensure` and `PyGILState_Release` are implemented in
  `module/thread` rather than here, and declared in `pystate.h` rather than the
  generated header, because a build without this layer exports them too — for
  cffi's embedding header;
- a `dict` subclass wherever `PyDict_Check` is the gate, which is every
  `PyDict_*` entry point and the keyword mapping of `PyObject_Call`. Such an
  instance is not a dict in pyre but an object holding one in a reserved layout
  slot (`typedef.rs dict_descr_new`), so each of them resolves that mapping and
  operates on it directly, consulting no override the subclass declares. A
  `list` subclass instance is a `W_ListObject` and needs none of this;
- `PyArg_ParseTuple`, `PyArg_ParseTupleAndKeywords`, `PyArg_UnpackTuple`,
  `Py_BuildValue`, `PyErr_Format` and the two `PyObject_Call*` pairs, whose
  bodies are C rather than Rust because rustc's `c_variadic` is unstable. They
  are compiled into the interpreter and exported from it
  (`pyre/pyre-interpreter/src/cpyext/src/`, the peer of
  `pypy/module/cpyext/src/getargs.c` and its siblings), so an extension
  resolves them at load time rather than compiling a copy of each;
- PyPy-style path-to-module-dictionary extension caching for single-phase
  modules — a multi-phase module is rebuilt from its definition on each import,
  as upstream's `create_cpyext_module` does;
- C-defined types (`cpyext/typeobject.rs`): `PyType_Ready` with `tp_base`
  inheritance and `inherit_slots`, `PyType_GenericAlloc` / `PyType_GenericNew` /
  `PyObject_Init`, `PyType_IsSubtype` / `PyType_GetFlags` / `PyObject_TypeCheck`,
  the `tp_methods` / `tp_members` / `tp_getset` descriptors with `__objclass__`,
  and wrappers for `tp_new`, `tp_init`, `tp_repr`, `tp_str`, `tp_hash`,
  `tp_call`, `tp_iter`, `tp_iternext`, `tp_richcompare`, `tp_getattro`,
  `tp_setattro`, `tp_descr_get` and `tp_descr_set`;
- heap types: `PyType_FromSpec`, `PyType_FromSpecWithBases`,
  `PyType_FromModuleAndSpec`, `PyType_GetSlot`, `PyType_GetName` and
  `PyType_GetQualName`, with every `slots_generated.h` identifier, `Py_tp_token` and
  `Py_tp_vectorcall` included -- the latter answering `Type(...)` ahead of
  `__new__`/`__init__` the way `PyType_Type`'s own
  `Py_TPFLAGS_HAVE_VECTORCALL` and `tp_vectorcall_offset` arrange for it;
- the `tp_as_number`, `tp_as_sequence` and `tp_as_mapping` tables, each slot
  becoming the dunder `slotdefs.py` names for it, and the `PyNumber_*`,
  `PySequence_*` and `PyMapping_*` entry points (`cpyext/number.rs`,
  `cpyext/sequence.rs`, `cpyext/mapping.rs`), which go through the
  interpreter's own operators so either operand may be a pyre object;
- the `tp_as_async` table as `__await__` / `__aiter__` / `__anext__`, with
  `am_send` reached through `PyIter_Send`, and the iterator entry points
  `PyObject_GetIter`, `PyObject_SelfIter`, `PyIter_Check`, `PyIter_Next`,
  `PyIter_NextItem`, `PyObject_GetAIter` and `PyAIter_Check`
  (`cpyext/iterator.rs`);
- the concrete `list`, `tuple` and `slice` protocols (`cpyext/listobject.rs`,
  `cpyext/tupleobject.rs`, `cpyext/sliceobject.rs`): all of `listobject.h` and
  `tupleobject.h`, and `PySlice_New` / `Check` / `Unpack` / `AdjustIndices` /
  `GetIndices` / `GetIndicesEx`.  Three of these are not exports:
  `PyTuple_Pack` is variadic, so it is a `static inline` in `Python.h` built
  from `PyTuple_New` and `PyTuple_SetItem`; `PySequence_Fast_GET_ITEM` and
  `PySequence_Fast_GET_SIZE` are functions rather than the macros the
  reference header spells, because a mirror carries no item array of its own
  and the length and items have to come from the interpreter object.
  `PySlice_GetIndicesEx` is both, matching the reference header: a macro over
  `PySlice_Unpack` and `PySlice_AdjustIndices` plus an export that is the same
  composition;
- the `tp_as_buffer` table (`cpyext/buffer.rs`): a C exporter's `bf_getbuffer`
  becomes a `memoryview` over the exported memory and its `bf_releasebuffer`
  runs when that view is released, so `memoryview(x)`, `bytes(x)` and the
  element writes all reach the exporter's own storage; in the other direction
  `PyObject_GetBuffer` / `PyBuffer_Release` / `PyBuffer_FillInfo` /
  `PyBuffer_IsContiguous` / `PyBuffer_SizeFromFormat` / `PyBuffer_GetPointer` /
  `PyBuffer_ToContiguous` / `PyBuffer_FromContiguous` / `PyObject_CopyData`, the
  legacy `PyObject_As*Buffer` family, and `PyMemoryView_FromObject` /
  `FromMemory` / `FromBuffer`;
- `PyErr_NewException` and `PyErr_NewExceptionWithDoc`, which build the class
  through the interpreter's own `type` so it gets the exception layout;
- capsules (`cpyext/capsule.rs`) and the strong-reference import entry points
  `PyImport_ImportModule`, `PyImport_Import`, `PyImport_AddModuleRef` and
  `PyImport_GetModule` (`cpyext/import_.rs`);
- GC forwarding for cached dictionaries, and `rawrefcount` in the collector
  itself (`majit-gc/src/rawrefcount.rs`, `majit-gc/src/collector.rs`): a mirror
  and its interpreter object are joined by a P-link, the collector traces and
  frees the two together in its own minor and major passes, and a mirror whose
  object died is handed back through a dead queue that `tp_dealloc` drains from
  an execution-context action. `tp_dealloc` and `tp_free` therefore run, and an
  instance of a C-defined type is reclaimed.
- the cyclic-collection protocol (`cpyext/gc.rs`): `PyObject_GC_Track` /
  `UnTrack` and the set of blocks that declare `Py_TPFLAGS_HAVE_GC`, whose
  `tp_traverse` reports what each one references. A collection is handed those
  edges before it traces anything, and reads them twice: a reference another
  block in the set supplies stops rooting on its own account, and it roots
  again once that block's own object is known to live -- which only the settled
  trace answers, and answering it before the trace frees objects that are
  alive. What survives neither test is a cycle, and `clear_garbage` runs the
  `tp_clear` that breaks it apart, the references being C fields no other layer
  can drop. `tp_traverse` is one of two sources of such a reference; the other
  is the borrow the layer owns on a container's behalf for `PyList_GetItem` and
  friends, which lives in a side table no traverse can reach and is reported
  alongside. Upstream has none of this; the shape follows CPython's
  `gcmodule.c`, and the one pypy precedent for calling a user traverse from a
  collection is `_hpy_universal`'s tracer.

The loader serializes extension initialization in the current import path;
the complete port must move that ownership into interpreter/execution-context
state before subinterpreters or parallel no-GIL imports are enabled.

Known divergences, each documented at its definition:

- a block whose type declares no `tp_traverse` reports no references, so a
  reference cycle running through one leaks: with nothing to subtract, a C
  reference marks its object as surviving along with everything the object
  reaches. That is where the whole feature stands upstream, `rawrefcount`
  having no pass that consults `tp_traverse` at all
  (`pypy/doc/discussion/rawrefcount.rst`); a type that declares the protocol is
  collected here;
- `md_def` and `md_state` ride reserved module-dictionary keys, and the
  `PyMethodDef` pointer, the descriptor definitions and a capsule's four C
  values ride reserved carrier-dictionary keys, because pyre has no typed
  payload for any of them yet;
- a capsule's destructor is recorded and never called, and the borrowed-
  reference `PyImport_AddModule` / `PyImport_GetModuleDict` are absent: pyre has
  no container for the borrow to belong to;
- the module state block is never released, pyre having no module deallocation
  path;
- `PyObject_GC_IsTracked` is the constant 1 and `PyObject_GC_IsFinalized` the
  constant 0: every object pyre holds is reachable by its collector, and
  nothing runs `tp_finalize`;
- `PyObject_Malloc` / `Calloc` / `Realloc` hand out blocks from the same census
  the mirror allocator uses, so `PyObject_Free` releases either kind; a block
  from the `PyMem_*` family is a different allocator's and must go back to
  `PyMem_Free`. A request smaller than a `PyObject` header is rounded up to
  one, because freeing a block clears that header;
- `PyList_New(n)` fills the slots with `None` rather than NULL, `PyTuple_New(n)`
  leaving them NULL as CPython does;
- the locale codec converts as UTF-8 rather than as whatever `LC_CTYPE`
  names: pyre reads no locale, and the filesystem encoding it does have is
  UTF-8 on the platforms this builds for. A process started under a non-UTF-8
  locale therefore converts as UTF-8 where CPython would follow the locale;
- `PyErr_SetRaisedException` and `PyErr_SetHandledException` refuse an object
  that is not an exception instance, with the `SystemError` `_PyErr_SetObject`
  uses for the same mistake. CPython checks neither, and a foreign object
  reaching its unwinder is a garbage read at the exception layout's offsets --
  measured as a spurious `SystemError` for an `int` and a segfault for a `str`
  -- so there is no behaviour there to match;
- `PyException_SetArgs` stores the items of its argument as `descr_setargs`
  does, so a non-tuple sequence reads back as the tuple of its items where
  CPython stores it verbatim: `args` is a typed slot here rather than a field
  holding whatever was last written to it;
- `PyErr_SetHandledException` takes no reference of its own, the handled slot
  being a collector root rather than a counted reference. A caller that
  releases the reference it handed over -- which the borrowing contract asks
  for -- still finds the slot valid, but the refcount it can read has not
  moved;
- a type is built on a single base, so a `PyType_Spec` naming more than one is
  rejected rather than silently losing the rest, and `PyType_FromMetaclass`
  accepts only `type` for its metaclass: pyre builds a type through its own
  constructor, which has no place to put another one;
- a frozen type is one whose `flag_heaptype` is false, the state every builtin
  type is already in and the one `type.__setattr__` consults. CPython keeps
  the two bits apart, so a type frozen here also stops answering
  `__annotations__`;
- `PyType_ClearCache` empties the method cache and reports 0: upstream's return
  value is a per-interpreter version counter, and pyre mints one version
  identity per type, so no single number describes the entries dropped;
- `PyObject_GetBuffer` of an *interpreter* object hands C a read-only snapshot,
  the collector being free to move the storage a `Py_buffer` would otherwise
  name; a `PyBUF_WRITABLE` request for one is refused rather than answered with
  a copy whose writes go nowhere. The `w*` argument format therefore reaches
  only a C exporter's own `bf_getbuffer`, and raises `BufferError` for a
  `bytearray`;
- `PyBytes_AS_STRING` and `PyBytes_GET_SIZE` are the calls their checked
  spellings make, and `PyUnicode_GET_LENGTH` is `PyUnicode_GetLength`: neither
  a `bytes` nor a `str` mirror carries the field the reference header reads,
  the storage being a cached copy rather than a tail allocated with the
  object;
- a C exporter's view keeps whatever geometry it declares -- strided,
  multi-dimensional, or indirect through `suboffsets`. Where a contiguous
  one-dimensional export becomes the same `RawBufferView` any in-heap exporter
  produces, everything else rides on a variant carrying its own `shape`,
  `strides` and `suboffsets`, ported from `CPyBuffer`
  (`pypy/module/cpyext/buffer.py`). Two deliberate deviations go with it, both
  toward CPython 3.14 and away from what pypy does today:
  - **suboffsets are honoured, where pypy declines them.** `BufferView.get_offset`
    carries `# TODO suboffsets?` at the `ADJUST_PTR` site, `_init_flags` guards
    `MEMORYVIEW_PIL` behind `if False`, and `w_get_suboffsets` answers `()`
    unconditionally. pyre implements `HAVE_PTR` / `ADJUST_PTR`, `init_suboffsets`
    and the `_Py_MEMORYVIEW_PIL` flag rule instead. An element behind a
    suboffset has no byte offset within the backing, so the address is chained
    dimension by dimension rather than summed, and the byte-slice bound the
    direct walk clamps against does not apply to it: past the first
    dereference the exporter's own pointers are the only description of the
    target, which is the trust CPython extends as well.
  - **a slice's length is `product(shape) * itemsize`, where `BufferSlice.getlength`
    is `shape[0] * itemsize`.** pypy has the same formula in-tree as
    `bytecount_from_shape`, but nothing reads it back; pyre folded
    `W_MemoryView.length` into the view, so the arm takes `init_len`'s answer.
  A window is computed for the storage a strided export addresses, because
  `Buffer::as_bytes` hands out a bounded slice where CPython and pypy both index
  a raw pointer: a negative stride reaches below `Py_buffer.buf`, so the block
  is lowered to the window's low end and the view's offset carries the shift.
  The `Py_buffer` a `bf_getbuffer` filled in is kept and handed back to
  `bf_releasebuffer` unchanged -- `internal` is the exporter's own state --
  keyed by the exported address, since the interpreter object it belongs to
  moves and the foreign memory does not;
- a `memoryview` that is never released never ends its export, pyre having no
  reference counting to end it at the last drop;
- the argument parser words a bad argument its own way: every message is
  prefixed with the function name `_PyPyre_ArgError` was given, where CPython
  numbers the argument instead and leaves the function out when the format
  carries no `:name`;
- `PyErr_BadInternalCall()` names the caller's own file and line, as its macro
  spelling does; a call made inside the runtime has no such place to name and
  reaches the plain entry point, so the same mistake reads differently
  depending on which side made it;
- `PyThread_acquire_lock_timed` never answers `PY_LOCK_INTR`: its wait is not
  interruptible, so a caller that loops on that status simply never goes round
  again;
- `PyErr_WarnExplicitFormat` defaults a NULL category to `RuntimeWarning`,
  which the entry point it shares its core with already does. CPython's is the
  one spelling in the family that skips that default and hands the NULL
  straight to `warn_explicit`, which calls it -- measured as a segfault -- so
  there is no behaviour there to match;
- three `rawrefcount` passes diverge from `incminimark.py` because the
  collector they sit in is not the one upstream wrote them for, each documented
  at its definition in `majit-gc/src/collector.rs`: the non-moving major
  additionally traces the *young* lists, that collection running no leading
  minor; a young mirror whose object was pinned stays young rather than being
  freed, a pinned survivor carrying no forwarding pointer to prove it alive;
  and a mirror linked to an object outside the managed heap is treated as
  alive, such an object having no header to read a mark from.

## What remains

The remaining generated API. Of the 746 public
`PyAPI_FUNC` entry points CPython 3.14.7 declares in its top-level
`Include/*.h` -- public meaning the declared name does not begin with an
underscore -- 460 are present, counting every form `Python.h` offers one
in: an export, a `static inline`, or a macro of either kind. (The
previously recorded 763/292 came from a pattern that missed the
declarations annotated `_Py_NO_RETURN` on one side and the object-like
aliases on the other. The 749/434 recorded after that was read with a
census script of its own; the population is now read with
`cpyext-abi.py`'s own declaration reader, the same one the gate uses,
which finds three fewer declarations in the same headers.) The figure
counts only that population, so the private `_PyLong_*ByteArray` pair and
the unchecked accessor macros the extensions below need do not appear in
it. The exported *data* is a separate population, counted and checked
separately (see below): 121 of the objects the header declares are
measured against CPython's own declaration for them;
6. Windows API DLL/import-library packaging.

`PyCFunction_Type` is the one `PyAPI_DATA(PyTypeObject)` name deliberately
absent. A method an extension defines carries `cpyext/methodobject.rs`'s own
`builtin_function_or_method`, which is a different type object from the one
`len` carries, so a single symbol cannot name the type of both. Binding it
either way would make `Py_IS_TYPE(op, &PyCFunction_Type)` answer wrongly for
half the objects it is asked about.

A type built from a spec carries the module it was created from and the
`Py_tp_token` it declared, both in a side table keyed by the type's own
address: pyre has no heap-type struct, and a spec type is leaked deliberately,
so the key is stable for the life of the process. The module is held there as
an owned mirror reference, and it is that reference — a count above the link
share — that roots it, not the table.

`PySequence_Fast_ITEMS` is absent rather than pending: it hands out a
`PyObject **` into the sequence's own storage, which requires a list whose
items are a contiguous array of mirrors. Upstream converts the list to a
CPython-backed strategy for exactly this (`cpyext/listobject.py`
`get_list_storage`); pyre has no such strategy, so there is no array to point
at.

The public suffix uses `pyre315`, not `cpython-315`: accepting a CPython-tagged
binary would falsely claim that CPython's private object layouts and symbols
are ABI-compatible with pyre.

## A published extension

`mmh3` 5.2.1 -- MurmurHash3 bindings, hand-written C, no code generator -- was
compiled against this header and run. It builds three static types through
`PyType_Ready`, publishes them with `PyModule_AddObject` from a single-phase
`PyInit_mmh3`, and reaches the interpreter through `METH_FASTCALL`,
`METH_NOARGS`, `METH_O` and `METH_VARARGS | METH_KEYWORDS` methods, `tp_new` /
`tp_init` / `tp_dealloc` / `tp_getset`, the `s*` argument format, and
`_PyLong_FromByteArray`. Its module-level functions and all three hashers --
one-shot, streaming across several `update` calls, and `copy` -- answer what
CPython 3.14.6 answers for the same inputs.

Nothing in the tree builds it: the sources are not vendored, and the fixtures
under `pyrex/tests/fixtures` cover the entry points it needed rather than the
package itself.

## PyPy reference map

- `pypy/module/cpyext/api.py`: API declaration/generation, C-call boundary,
  extension loader and initialization checks.
- `pypy/module/cpyext/pyobject.py`: raw mirror descriptors and W_Root ↔
  `PyObject *` conversion.
- `rpython/rlib/rawrefcount.py` and `pypy/doc/discussion/rawrefcount.rst`: the
  vocabulary -- the P/O link kinds, `REFCNT_FROM_PYPY`, the dead queue -- and
  what a C reference does and does not keep alive.
- `rpython/memory/gc/incminimark.py`, the `rrc_*` methods: the algorithm
  itself. The front end declares; the collector owns the lists, the traces and
  the frees.
- `pypy/module/cpyext/state.py`: exception state, extension cache and GC dead
  queue.
- `pypy/module/cpyext/modsupport.py`: module definitions, method conversion,
  PEP 489 slots and the `PyModule_*` entry points. `PyModule_GetName` /
  `GetNameObject` read `__name__` from the module dictionary where
  `modsupport.py:240-276` reads `w_mod.w_name`, so a module renamed after
  import reports the new name; `PyModule_GetFilenameObject` reports a missing
  or non-`str` `__file__` as the `SystemError` it is documented to raise
  rather than as the `KeyError` a plain `getitem` produces.
- `pypy/module/cpyext/methodobject.py`: the `PyCFunction` carrier and its
  calling conventions.
- `pypy/module/cpyext/pyerrors.py`: the `PyErr_*` entry points.
- `pypy/module/cpyext/exception.py`: the `PyException_*` slot accessors.
- `pypy/module/cpyext/typeobject.py`: `PyType_Ready`, slot inheritance and the
  descriptors built from `tp_methods` / `tp_members` / `tp_getset`.
- `pypy/module/cpyext/slotdefs.py`: the wrappers that turn a C slot into an
  app-level method.
- `pypy/module/cpyext/structmemberdefs.py`: the `T_*` member type codes.
- `pypy/module/cpyext/number.py`, `sequence.py` and `mapping.py`: the
  `PyNumber_*`, `PySequence_*` and `PyMapping_*` entry points.
- `pypy/module/cpyext/object.py`: the object protocol, the object allocator and
  the `object.__getattribute__` / `__setattr__` / `__dict__` terminals.
  `PyObject_Bytes` runs a `__bytes__` override and then falls back to
  `PyBytes_FromObject`, which does not consult it -- so the two answer
  differently for an object that defines one. `PyObject_Hash` reports a hash of
  -1 as -2, -1 being the failure answer.
- `pypy/module/cpyext/longobject.py`: the `int` conversions. `PyLong_AsInt32` /
  `AsUInt32` / `AsInt64` / `AsUInt64` and their `From` counterparts are 3.14
  additions upstream predates, as are `PyLong_AsNativeBytes`,
  `PyLong_FromNativeBytes` and `PyLong_FromUnsignedNativeBytes`;
  `PyLong_GetInfo` reads `sys.int_info`, which is where the same numbers are
  already published.
- `pypy/module/cpyext/pycapsule.py` and `import_.py`: capsules and the import
  entry points.
- `pypy/module/cpyext/buffer.py` and `memoryobject.py`: `Py_buffer`, the
  `PyBuffer_*` entry points and `PyMemoryView_*`.
- `pypy/module/cpyext/iterator.py`: the iterator entry points.
- `pypy/module/cpyext/listobject.py`, `tupleobject.py` and `sliceobject.py`:
  the concrete list, tuple and slice entry points. Two divergences, both
  following the 3.14 specification where upstream predates it:
  `PySequence_Fast` answers a list where `sequence.py:52-66` answers a tuple,
  and `PyNumber_ToBase(n, 10)` is the plain decimal spelling rather than a
  radix format, `format_index_radix` covering only the power-of-two radices.
- `pypy/module/cpyext/src/getargs.c`: `PyArg_ParseTuple` and `Py_BuildValue`,
  which are C there too.
- `pypy/module/imp/interp_imp.py`: `_imp` entry points.
