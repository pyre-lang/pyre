/* The exported entry points, one block per `cpyext` module.
 *
 * Written by scripts/cpyext-abi.py generate; do not edit by hand.
 * A declaration here is CPython's own where CPython has one, so an
 * extension's prototypes and pyre's agree by construction.
 *
 * An export a hand-written header renames to an inline fast path is
 * left out: that header declares it ahead of the rename, which a
 * declaration here would come after.  So is one a header under
 * `internal/` declares, which is reached by including that header
 * and not by including `Python.h`.
 */
#ifndef PYRE_DECL_H
#define PYRE_DECL_H

#ifdef __cplusplus
extern "C" {
#endif

/* cpyext/buffer.rs */
PyAPI_FUNC(int) PyBuffer_FillInfo(Py_buffer *, PyObject *, void *, Py_ssize_t, int, int);
PyAPI_FUNC(int) PyBuffer_FromContiguous(const Py_buffer *, const void *, Py_ssize_t, char);
PyAPI_FUNC(void *) PyBuffer_GetPointer(const Py_buffer *, const Py_ssize_t *);
PyAPI_FUNC(int) PyBuffer_IsContiguous(const Py_buffer *, char);
PyAPI_FUNC(void) PyBuffer_Release(Py_buffer *);
PyAPI_FUNC(Py_ssize_t) PyBuffer_SizeFromFormat(const char *);
PyAPI_FUNC(int) PyBuffer_ToContiguous(void *, const Py_buffer *, Py_ssize_t, char);
PyAPI_FUNC(int) PyMemoryView_Check(PyObject *);
PyAPI_FUNC(PyObject *) PyMemoryView_FromBuffer(const Py_buffer *);
PyAPI_FUNC(PyObject *) PyMemoryView_FromMemory(char *, Py_ssize_t, int);
PyAPI_FUNC(PyObject *) PyMemoryView_FromObject(PyObject *);
PyAPI_FUNC(PyObject *) PyMemoryView_GetContiguous(PyObject *, int, char);
PyAPI_FUNC(int) PyObject_AsCharBuffer(PyObject *, const char **, Py_ssize_t *);
PyAPI_FUNC(int) PyObject_AsReadBuffer(PyObject *, const void **, Py_ssize_t *);
PyAPI_FUNC(int) PyObject_AsWriteBuffer(PyObject *, void **, Py_ssize_t *);
PyAPI_FUNC(int) PyObject_CheckBuffer(PyObject *);
PyAPI_FUNC(int) PyObject_CheckReadBuffer(PyObject *);
PyAPI_FUNC(int) PyObject_CopyData(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_GetBuffer(PyObject *, Py_buffer *, int);

/* cpyext/bytearrayobject.rs */
PyAPI_FUNC(char *) PyByteArray_AsString(PyObject *);
PyAPI_FUNC(int) PyByteArray_Check(PyObject *);
PyAPI_FUNC(int) PyByteArray_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyByteArray_Concat(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyByteArray_FromObject(PyObject *);
PyAPI_FUNC(PyObject *) PyByteArray_FromStringAndSize(const char *, Py_ssize_t);
PyAPI_FUNC(int) PyByteArray_Resize(PyObject *, Py_ssize_t);
PyAPI_FUNC(Py_ssize_t) PyByteArray_Size(PyObject *);

/* cpyext/bytesobject.rs */
PyAPI_FUNC(char *) PyBytes_AS_STRING(void *);
PyAPI_FUNC(char *) PyBytes_AsString(PyObject *);
PyAPI_FUNC(int) PyBytes_AsStringAndSize(PyObject *, char **, Py_ssize_t *);
PyAPI_FUNC(int) PyBytes_Check(PyObject *);
PyAPI_FUNC(int) PyBytes_CheckExact(PyObject *);
PyAPI_FUNC(void) PyBytes_Concat(PyObject **, PyObject *);
PyAPI_FUNC(void) PyBytes_ConcatAndDel(PyObject **, PyObject *);
PyAPI_FUNC(PyObject *) PyBytes_FromObject(PyObject *);
PyAPI_FUNC(PyObject *) PyBytes_FromString(const char *);
PyAPI_FUNC(PyObject *) PyBytes_FromStringAndSize(const char *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyBytes_Join(PyObject *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyBytes_Size(PyObject *);
PyAPI_FUNC(int) _PyBytes_Resize(PyObject **, Py_ssize_t);

/* cpyext/capsule.rs */
PyAPI_FUNC(int) PyCapsule_CheckExact(PyObject *);
PyAPI_FUNC(void *) PyCapsule_GetContext(PyObject *);
PyAPI_FUNC(PyCapsule_Destructor) PyCapsule_GetDestructor(PyObject *);
PyAPI_FUNC(const char *) PyCapsule_GetName(PyObject *);
PyAPI_FUNC(void *) PyCapsule_GetPointer(PyObject *, const char *);
PyAPI_FUNC(void *) PyCapsule_Import(const char *, int);
PyAPI_FUNC(int) PyCapsule_IsValid(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyCapsule_New(void *, const char *, PyCapsule_Destructor);
PyAPI_FUNC(int) PyCapsule_SetContext(PyObject *, void *);
PyAPI_FUNC(int) PyCapsule_SetDestructor(PyObject *, PyCapsule_Destructor);
PyAPI_FUNC(int) PyCapsule_SetName(PyObject *, const char *);
PyAPI_FUNC(int) PyCapsule_SetPointer(PyObject *, void *);

/* cpyext/cdatetime.rs */
PyAPI_FUNC(int) PyDateTime_Check(PyObject *);
PyAPI_FUNC(int) PyDateTime_CheckExact(PyObject *);
PyAPI_FUNC(int) PyDateTime_DATE_GET_HOUR(void *);
PyAPI_FUNC(int) PyDateTime_DATE_GET_MICROSECOND(void *);
PyAPI_FUNC(int) PyDateTime_DATE_GET_MINUTE(void *);
PyAPI_FUNC(int) PyDateTime_DATE_GET_SECOND(void *);
PyAPI_FUNC(PyObject *) PyDateTime_DATE_GET_TZINFO(void *);
PyAPI_FUNC(int) PyDateTime_DELTA_GET_DAYS(void *);
PyAPI_FUNC(int) PyDateTime_DELTA_GET_MICROSECONDS(void *);
PyAPI_FUNC(int) PyDateTime_DELTA_GET_SECONDS(void *);
PyAPI_FUNC(PyObject *) PyDateTime_FromTimestamp(PyObject *);
PyAPI_FUNC(int) PyDateTime_GET_DAY(void *);
PyAPI_FUNC(int) PyDateTime_GET_FOLD(void *);
PyAPI_FUNC(int) PyDateTime_GET_MONTH(void *);
PyAPI_FUNC(int) PyDateTime_GET_YEAR(void *);
PyAPI_FUNC(int) PyDateTime_TIME_GET_FOLD(void *);
PyAPI_FUNC(int) PyDateTime_TIME_GET_HOUR(void *);
PyAPI_FUNC(int) PyDateTime_TIME_GET_MICROSECOND(void *);
PyAPI_FUNC(int) PyDateTime_TIME_GET_MINUTE(void *);
PyAPI_FUNC(int) PyDateTime_TIME_GET_SECOND(void *);
PyAPI_FUNC(PyObject *) PyDateTime_TIME_GET_TZINFO(void *);
PyAPI_FUNC(int) PyDate_Check(PyObject *);
PyAPI_FUNC(int) PyDate_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyDate_FromTimestamp(PyObject *);
PyAPI_FUNC(int) PyDelta_Check(PyObject *);
PyAPI_FUNC(int) PyDelta_CheckExact(PyObject *);
PyAPI_FUNC(int) PyTZInfo_Check(PyObject *);
PyAPI_FUNC(int) PyTZInfo_CheckExact(PyObject *);
PyAPI_FUNC(int) PyTime_Check(PyObject *);
PyAPI_FUNC(int) PyTime_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) _PyDateTime_FromDateAndTime(int, int, int, int, int, int, int, PyObject *, PyTypeObject *);
PyAPI_FUNC(PyObject *) _PyDateTime_FromDateAndTimeAndFold(int, int, int, int, int, int, int, PyObject *, int, PyTypeObject *);
PyAPI_FUNC(PyObject *) _PyDateTime_FromTimestamp(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(PyDateTime_CAPI *) _PyDateTime_Import(void);
PyAPI_FUNC(PyObject *) _PyDate_FromDate(int, int, int, PyTypeObject *);
PyAPI_FUNC(PyObject *) _PyDate_FromTimestamp(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) _PyDelta_FromDelta(int, int, int, int, PyTypeObject *);
PyAPI_FUNC(PyObject *) _PyTimeZone_FromTimeZone(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) _PyTime_FromTime(int, int, int, int, PyObject *, PyTypeObject *);
PyAPI_FUNC(PyObject *) _PyTime_FromTimeAndFold(int, int, int, int, PyObject *, int, PyTypeObject *);

/* cpyext/classobject.rs */
PyAPI_FUNC(PyObject *) PyInstanceMethod_Function(PyObject *);
PyAPI_FUNC(PyObject *) PyInstanceMethod_New(PyObject *);

/* cpyext/codecs.rs */
PyAPI_FUNC(PyObject *) PyCodec_Decode(PyObject *, const char *, const char *);
PyAPI_FUNC(PyObject *) PyCodec_Decoder(const char *);
PyAPI_FUNC(PyObject *) PyCodec_Encode(PyObject *, const char *, const char *);
PyAPI_FUNC(PyObject *) PyCodec_Encoder(const char *);
PyAPI_FUNC(PyObject *) PyCodec_IncrementalDecoder(const char *, const char *);
PyAPI_FUNC(PyObject *) PyCodec_IncrementalEncoder(const char *, const char *);

/* cpyext/complexobject.rs */
PyAPI_FUNC(Py_complex) PyComplex_AsCComplex(PyObject *);
PyAPI_FUNC(int) PyComplex_Check(PyObject *);
PyAPI_FUNC(int) PyComplex_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyComplex_FromCComplex(Py_complex);
PyAPI_FUNC(PyObject *) PyComplex_FromDoubles(double, double);
PyAPI_FUNC(double) PyComplex_ImagAsDouble(PyObject *);
PyAPI_FUNC(double) PyComplex_RealAsDouble(PyObject *);

/* cpyext/contextvars.rs */
PyAPI_FUNC(int) PyContextVar_Get(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(PyObject *) PyContextVar_New(const char *, PyObject *);
PyAPI_FUNC(int) PyContextVar_Reset(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyContextVar_Set(PyObject *, PyObject *);

/* cpyext/dictobject.rs */
PyAPI_FUNC(PyObject *) PyDictProxy_New(PyObject *);
PyAPI_FUNC(int) PyDict_Check(PyObject *);
PyAPI_FUNC(int) PyDict_CheckExact(PyObject *);
PyAPI_FUNC(void) PyDict_Clear(PyObject *);
PyAPI_FUNC(int) PyDict_Contains(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyDict_Copy(PyObject *);
PyAPI_FUNC(int) PyDict_DelItem(PyObject *, PyObject *);
PyAPI_FUNC(int) PyDict_DelItemString(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyDict_GetItem(PyObject *, PyObject *);
PyAPI_FUNC(int) PyDict_GetItemRef(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(PyObject *) PyDict_GetItemString(PyObject *, const char *);
PyAPI_FUNC(int) PyDict_GetItemStringRef(PyObject *, const char *, PyObject **);
PyAPI_FUNC(PyObject *) PyDict_GetItemWithError(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyDict_Items(PyObject *);
PyAPI_FUNC(PyObject *) PyDict_Keys(PyObject *);
PyAPI_FUNC(int) PyDict_Merge(PyObject *, PyObject *, int);
PyAPI_FUNC(int) PyDict_MergeFromSeq2(PyObject *, PyObject *, int);
PyAPI_FUNC(PyObject *) PyDict_New(void);
PyAPI_FUNC(int) PyDict_Next(PyObject *, Py_ssize_t *, PyObject **, PyObject **);
PyAPI_FUNC(int) PyDict_Pop(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(int) PyDict_PopString(PyObject *, const char *, PyObject **);
PyAPI_FUNC(PyObject *) PyDict_SetDefault(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PyDict_SetDefaultRef(PyObject *, PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(int) PyDict_SetItem(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PyDict_SetItemString(PyObject *, const char *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyDict_Size(PyObject *);
PyAPI_FUNC(int) PyDict_Update(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyDict_Values(PyObject *);

/* cpyext/eval.rs */
PyAPI_FUNC(PyObject *) PyEval_GetBuiltins(void);
PyAPI_FUNC(int) Py_EnterRecursiveCall(const char *);
PyAPI_FUNC(void) Py_LeaveRecursiveCall(void);

/* cpyext/exception.rs */
PyAPI_FUNC(int) PyExceptionClass_Check(PyObject *);
PyAPI_FUNC(const char *) PyExceptionClass_Name(PyObject *);
PyAPI_FUNC(int) PyExceptionInstance_Check(PyObject *);
PyAPI_FUNC(PyObject *) PyException_GetArgs(PyObject *);
PyAPI_FUNC(PyObject *) PyException_GetCause(PyObject *);
PyAPI_FUNC(PyObject *) PyException_GetContext(PyObject *);
PyAPI_FUNC(PyObject *) PyException_GetTraceback(PyObject *);
PyAPI_FUNC(void) PyException_SetArgs(PyObject *, PyObject *);
PyAPI_FUNC(void) PyException_SetCause(PyObject *, PyObject *);
PyAPI_FUNC(void) PyException_SetContext(PyObject *, PyObject *);
PyAPI_FUNC(int) PyException_SetTraceback(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyUnicodeDecodeError_Create(const char *, const char *, Py_ssize_t, Py_ssize_t, Py_ssize_t, const char *);

/* cpyext/floatobject.rs */
PyAPI_FUNC(double) PyFloat_AS_DOUBLE(PyObject *);
PyAPI_FUNC(double) PyFloat_AsDouble(PyObject *);
PyAPI_FUNC(int) PyFloat_Check(PyObject *);
PyAPI_FUNC(int) PyFloat_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyFloat_FromDouble(double);
PyAPI_FUNC(PyObject *) PyFloat_FromString(PyObject *);
PyAPI_FUNC(int) PyFloat_Pack2(double, char *, int);
PyAPI_FUNC(int) PyFloat_Pack4(double, char *, int);
PyAPI_FUNC(int) PyFloat_Pack8(double, char *, int);
PyAPI_FUNC(double) PyFloat_Unpack2(const char *, int);
PyAPI_FUNC(double) PyFloat_Unpack4(const char *, int);
PyAPI_FUNC(double) PyFloat_Unpack8(const char *, int);

/* cpyext/frameobject.rs */
PyAPI_FUNC(PyFrameObject *) PyFrame_New(PyThreadState *, PyCodeObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PyTraceBack_Here(PyFrameObject *);

/* cpyext/funcobject.rs */
PyAPI_FUNC(PyCodeObject *) PyCode_NewEmpty(const char *, const char *, int);
PyAPI_FUNC(PyObject *) PyMethod_Function(PyObject *);
PyAPI_FUNC(PyObject *) PyMethod_New(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyMethod_Self(PyObject *);
PyAPI_FUNC(PyCodeObject *) PyUnstable_Code_New(int, int, int, int, int, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, int, PyObject *, PyObject *);
PyAPI_FUNC(PyCodeObject *) PyUnstable_Code_NewWithPosOnlyArgs(int, int, int, int, int, int, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, PyObject *, int, PyObject *, PyObject *);

/* cpyext/gc.rs */
PyAPI_FUNC(void) PyObject_GC_Track(void *);
PyAPI_FUNC(void) PyObject_GC_UnTrack(void *);

/* cpyext/genericaliasobject.rs */
PyAPI_FUNC(PyObject *) Py_GenericAlias(PyObject *, PyObject *);

/* cpyext/import_.rs */
PyAPI_FUNC(PyObject *) PyImport_AddModuleRef(const char *);
PyAPI_FUNC(PyObject *) PyImport_GetModule(PyObject *);
PyAPI_FUNC(PyObject *) PyImport_GetModuleDict(void);
PyAPI_FUNC(PyObject *) PyImport_Import(PyObject *);
PyAPI_FUNC(PyObject *) PyImport_ImportModule(const char *);
PyAPI_FUNC(PyObject *) PyImport_ImportModuleAttr(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyImport_ImportModuleAttrString(const char *, const char *);
PyAPI_FUNC(PyObject *) PyImport_ImportModuleLevel(const char *, PyObject *, PyObject *, PyObject *, int);
PyAPI_FUNC(PyObject *) PyImport_ImportModuleLevelObject(PyObject *, PyObject *, PyObject *, PyObject *, int);
PyAPI_FUNC(PyObject *) PyImport_ImportModuleNoBlock(const char *);

/* cpyext/iterator.rs */
PyAPI_FUNC(int) PyAIter_Check(PyObject *);
PyAPI_FUNC(int) PyIter_Check(PyObject *);
PyAPI_FUNC(PyObject *) PyIter_Next(PyObject *);
PyAPI_FUNC(int) PyIter_NextItem(PyObject *, PyObject **);
PyAPI_FUNC(PySendResult) PyIter_Send(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(PyObject *) PyObject_GetAIter(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_GetIter(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_SelfIter(PyObject *);

/* cpyext/listobject.rs */
PyAPI_FUNC(int) PyList_Append(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyList_AsTuple(PyObject *);
PyAPI_FUNC(int) PyList_Check(PyObject *);
PyAPI_FUNC(int) PyList_CheckExact(PyObject *);
PyAPI_FUNC(int) PyList_Clear(PyObject *);
PyAPI_FUNC(int) PyList_Extend(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyList_GetItem(PyObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyList_GetItemRef(PyObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyList_GetSlice(PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(int) PyList_Insert(PyObject *, Py_ssize_t, PyObject *);
PyAPI_FUNC(PyObject *) PyList_New(Py_ssize_t);
PyAPI_FUNC(int) PyList_Reverse(PyObject *);
PyAPI_FUNC(int) PyList_SetItem(PyObject *, Py_ssize_t, PyObject *);
PyAPI_FUNC(int) PyList_SetSlice(PyObject *, Py_ssize_t, Py_ssize_t, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyList_Size(PyObject *);
PyAPI_FUNC(int) PyList_Sort(PyObject *);

/* cpyext/lock.rs */
PyAPI_FUNC(void) PyCriticalSection_Begin(PyCriticalSection *, PyObject *);
PyAPI_FUNC(void) PyCriticalSection_End(PyCriticalSection *);
PyAPI_FUNC(int) PyThread_acquire_lock(PyThread_type_lock, int);
PyAPI_FUNC(PyLockStatus) PyThread_acquire_lock_timed(PyThread_type_lock, PY_TIMEOUT_T, int);
PyAPI_FUNC(PyThread_type_lock) PyThread_allocate_lock(void);
PyAPI_FUNC(void) PyThread_free_lock(PyThread_type_lock);
PyAPI_FUNC(unsigned long) PyThread_get_thread_ident(void);
PyAPI_FUNC(void) PyThread_release_lock(PyThread_type_lock);

/* cpyext/longobject.rs */
PyAPI_FUNC(PyObject *) PyBool_FromLong(long);
PyAPI_FUNC(double) PyLong_AsDouble(PyObject *);
PyAPI_FUNC(int) PyLong_AsInt(PyObject *);
PyAPI_FUNC(int) PyLong_AsInt32(PyObject *, int32_t *);
PyAPI_FUNC(int) PyLong_AsInt64(PyObject *, int64_t *);
PyAPI_FUNC(long) PyLong_AsLong(PyObject *);
PyAPI_FUNC(long) PyLong_AsLongAndOverflow(PyObject *, int *);
PyAPI_FUNC(long long) PyLong_AsLongLong(PyObject *);
PyAPI_FUNC(long long) PyLong_AsLongLongAndOverflow(PyObject *, int *);
PyAPI_FUNC(Py_ssize_t) PyLong_AsNativeBytes(PyObject *, void *, Py_ssize_t, int);
PyAPI_FUNC(size_t) PyLong_AsSize_t(PyObject *);
PyAPI_FUNC(Py_ssize_t) PyLong_AsSsize_t(PyObject *);
PyAPI_FUNC(int) PyLong_AsUInt32(PyObject *, uint32_t *);
PyAPI_FUNC(int) PyLong_AsUInt64(PyObject *, uint64_t *);
PyAPI_FUNC(unsigned long) PyLong_AsUnsignedLong(PyObject *);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLong(PyObject *);
PyAPI_FUNC(unsigned long long) PyLong_AsUnsignedLongLongMask(PyObject *);
PyAPI_FUNC(unsigned long) PyLong_AsUnsignedLongMask(PyObject *);
PyAPI_FUNC(void *) PyLong_AsVoidPtr(PyObject *);
PyAPI_FUNC(int) PyLong_Check(PyObject *);
PyAPI_FUNC(int) PyLong_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyLong_FromDouble(double);
PyAPI_FUNC(PyObject *) PyLong_FromInt32(int32_t);
PyAPI_FUNC(PyObject *) PyLong_FromInt64(int64_t);
PyAPI_FUNC(PyObject *) PyLong_FromLong(long);
PyAPI_FUNC(PyObject *) PyLong_FromLongLong(long long);
PyAPI_FUNC(PyObject *) PyLong_FromNativeBytes(const void *, size_t, int);
PyAPI_FUNC(PyObject *) PyLong_FromSize_t(size_t);
PyAPI_FUNC(PyObject *) PyLong_FromSsize_t(Py_ssize_t);
PyAPI_FUNC(PyObject *) PyLong_FromString(const char *, char **, int);
PyAPI_FUNC(PyObject *) PyLong_FromUInt32(uint32_t);
PyAPI_FUNC(PyObject *) PyLong_FromUInt64(uint64_t);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLong(unsigned long);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedLongLong(unsigned long long);
PyAPI_FUNC(PyObject *) PyLong_FromUnsignedNativeBytes(const void *, size_t, int);
PyAPI_FUNC(PyObject *) PyLong_FromVoidPtr(void *);
PyAPI_FUNC(PyObject *) PyLong_GetInfo(void);
PyAPI_FUNC(PyObject *) PyNumber_Long(PyObject *);
PyAPI_FUNC(int) _PyLong_AsByteArray(PyLongObject *, unsigned char *, size_t, int, int, int);
PyAPI_FUNC(PyObject *) _PyLong_FromByteArray(const unsigned char *, size_t, int, int);
PyAPI_FUNC(int) _PyLong_Sign(PyObject *);

/* cpyext/mapping.rs */
PyAPI_FUNC(int) PyMapping_Check(PyObject *);
PyAPI_FUNC(int) PyMapping_DelItemString(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyMapping_GetItemString(PyObject *, const char *);
PyAPI_FUNC(int) PyMapping_GetOptionalItem(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(int) PyMapping_GetOptionalItemString(PyObject *, const char *, PyObject **);
PyAPI_FUNC(int) PyMapping_HasKey(PyObject *, PyObject *);
PyAPI_FUNC(int) PyMapping_HasKeyString(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyMapping_Items(PyObject *);
PyAPI_FUNC(PyObject *) PyMapping_Keys(PyObject *);
PyAPI_FUNC(Py_ssize_t) PyMapping_Length(PyObject *);
PyAPI_FUNC(int) PyMapping_SetItemString(PyObject *, const char *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyMapping_Size(PyObject *);
PyAPI_FUNC(PyObject *) PyMapping_Values(PyObject *);

/* cpyext/methodobject.rs */
PyAPI_FUNC(int) PyCFunction_GetFlags(PyObject *);
PyAPI_FUNC(PyCFunction) PyCFunction_GetFunction(PyObject *);
PyAPI_FUNC(PyObject *) PyCFunction_GetSelf(PyObject *);
PyAPI_FUNC(PyObject *) PyCFunction_New(PyMethodDef *, PyObject *);
PyAPI_FUNC(PyObject *) PyCFunction_NewEx(PyMethodDef *, PyObject *, PyObject *);
PyAPI_FUNC(PyTypeObject *) PyCMethod_GetClass(PyObject *);
PyAPI_FUNC(PyObject *) PyCMethod_New(PyMethodDef *, PyObject *, PyObject *, PyTypeObject *);
PyAPI_FUNC(PyObject *) PyClassMethod_New(PyObject *);
PyAPI_FUNC(PyObject *) PyDescr_NewClassMethod(PyTypeObject *, PyMethodDef *);
PyAPI_FUNC(PyObject *) PyDescr_NewMethod(PyTypeObject *, PyMethodDef *);
PyAPI_FUNC(int) PyMethodDescr_Check(PyObject *);
PyAPI_FUNC(int) PyMethodDescr_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyStaticMethod_New(PyObject *);

/* cpyext/modsupport.rs */
PyAPI_FUNC(PyObject *) PyModuleDef_Init(PyModuleDef *);
PyAPI_FUNC(int) PyModule_Add(PyObject *, const char *, PyObject *);
PyAPI_FUNC(int) PyModule_AddFunctions(PyObject *, PyMethodDef *);
PyAPI_FUNC(int) PyModule_AddIntConstant(PyObject *, const char *, long);
PyAPI_FUNC(int) PyModule_AddObject(PyObject *, const char *, PyObject *);
PyAPI_FUNC(int) PyModule_AddObjectRef(PyObject *, const char *, PyObject *);
PyAPI_FUNC(int) PyModule_AddStringConstant(PyObject *, const char *, const char *);
PyAPI_FUNC(int) PyModule_Check(PyObject *);
PyAPI_FUNC(int) PyModule_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyModule_Create2(PyModuleDef *, int);
PyAPI_FUNC(int) PyModule_ExecDef(PyObject *, PyModuleDef *);
PyAPI_FUNC(PyObject *) PyModule_FromDefAndSpec2(PyModuleDef *, PyObject *, int);
PyAPI_FUNC(PyModuleDef *) PyModule_GetDef(PyObject *);
PyAPI_FUNC(PyObject *) PyModule_GetDict(PyObject *);
PyAPI_FUNC(const char *) PyModule_GetFilename(PyObject *);
PyAPI_FUNC(PyObject *) PyModule_GetFilenameObject(PyObject *);
PyAPI_FUNC(const char *) PyModule_GetName(PyObject *);
PyAPI_FUNC(PyObject *) PyModule_GetNameObject(PyObject *);
PyAPI_FUNC(void *) PyModule_GetState(PyObject *);
PyAPI_FUNC(PyObject *) PyModule_New(const char *);
PyAPI_FUNC(PyObject *) PyModule_NewObject(PyObject *);
PyAPI_FUNC(int) PyModule_SetDocString(PyObject *, const char *);
PyAPI_FUNC(int) PyState_AddModule(PyObject *, PyModuleDef *);
PyAPI_FUNC(PyObject *) PyState_FindModule(PyModuleDef *);
PyAPI_FUNC(int) PyState_RemoveModule(PyModuleDef *);

/* cpyext/number.rs */
PyAPI_FUNC(int) PyIndex_Check(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Absolute(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Add(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_And(PyObject *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyNumber_AsSsize_t(PyObject *, PyObject *);
PyAPI_FUNC(int) PyNumber_Check(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Divmod(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Float(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_FloorDivide(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceAdd(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceAnd(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceFloorDivide(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceLshift(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceMatrixMultiply(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceMultiply(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceOr(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlacePower(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceRemainder(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceRshift(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceSubtract(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceTrueDivide(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_InPlaceXor(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Index(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Invert(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Lshift(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_MatrixMultiply(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Multiply(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Negative(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Or(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Positive(PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Power(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Remainder(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Rshift(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Subtract(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_ToBase(PyObject *, int);
PyAPI_FUNC(PyObject *) PyNumber_TrueDivide(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyNumber_Xor(PyObject *, PyObject *);

/* cpyext/object.rs */
PyAPI_FUNC(int) PyCallable_Check(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_ASCII(PyObject *);
PyAPI_FUNC(int) PyObject_AsFileDescriptor(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_Bytes(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_Call(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(void) PyObject_CallFinalizer(PyObject *);
PyAPI_FUNC(int) PyObject_CallFinalizerFromDealloc(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_CallNoArgs(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_CallObject(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyObject_CallOneArg(PyObject *, PyObject *);
PyAPI_FUNC(void *) PyObject_Calloc(size_t, size_t);
PyAPI_FUNC(void) PyObject_ClearManagedDict(PyObject *);
PyAPI_FUNC(int) PyObject_DelAttr(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_DelAttrString(PyObject *, const char *);
PyAPI_FUNC(int) PyObject_DelItem(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_DelItemString(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyObject_Dir(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_Format(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_GC_IsFinalized(PyObject *);
PyAPI_FUNC(int) PyObject_GC_IsTracked(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_GenericGetAttr(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyObject_GenericGetDict(PyObject *, void *);
PyAPI_FUNC(int) PyObject_GenericSetAttr(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_GenericSetDict(PyObject *, PyObject *, void *);
PyAPI_FUNC(PyObject *) PyObject_GetAttr(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyObject_GetAttrString(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyObject_GetItem(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_GetOptionalAttr(PyObject *, PyObject *, PyObject **);
PyAPI_FUNC(int) PyObject_GetOptionalAttrString(PyObject *, const char *, PyObject **);
PyAPI_FUNC(int) PyObject_HasAttr(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_HasAttrString(PyObject *, const char *);
PyAPI_FUNC(int) PyObject_HasAttrStringWithError(PyObject *, const char *);
PyAPI_FUNC(int) PyObject_HasAttrWithError(PyObject *, PyObject *);
PyAPI_FUNC(Py_hash_t) PyObject_Hash(PyObject *);
PyAPI_FUNC(Py_hash_t) PyObject_HashNotImplemented(PyObject *);
PyAPI_FUNC(PyVarObject *) PyObject_InitVar(PyVarObject *, PyTypeObject *, Py_ssize_t);
PyAPI_FUNC(int) PyObject_IsInstance(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_IsSubclass(PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_IsTrue(PyObject *);
PyAPI_FUNC(Py_ssize_t) PyObject_Length(PyObject *);
PyAPI_FUNC(void *) PyObject_Malloc(size_t);
PyAPI_FUNC(int) PyObject_Not(PyObject *);
PyAPI_FUNC(void *) PyObject_Realloc(void *, size_t);
PyAPI_FUNC(PyObject *) PyObject_Repr(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_RichCompare(PyObject *, PyObject *, int);
PyAPI_FUNC(int) PyObject_RichCompareBool(PyObject *, PyObject *, int);
PyAPI_FUNC(int) PyObject_SetAttr(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PyObject_SetAttrString(PyObject *, const char *, PyObject *);
PyAPI_FUNC(int) PyObject_SetItem(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyObject_Size(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_Str(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_Type(PyObject *);
PyAPI_FUNC(PyObject *) PyObject_Vectorcall(PyObject *, PyObject *const *, size_t, PyObject *);
PyAPI_FUNC(PyObject *) PyObject_VectorcallDict(PyObject *, PyObject *const *, size_t, PyObject *);
PyAPI_FUNC(PyObject *) PyObject_VectorcallMethod(PyObject *, PyObject *const *, size_t, PyObject *);
PyAPI_FUNC(int) PyObject_VisitManagedDict(PyObject *, visitproc, void *);
PyAPI_FUNC(PyObject *) PyVectorcall_Call(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) Py_GetConstant(unsigned int);
PyAPI_FUNC(PyObject *) Py_GetConstantBorrowed(unsigned int);
PyAPI_FUNC(Py_hash_t) Py_HashBuffer(const void *, Py_ssize_t);
PyAPI_FUNC(Py_hash_t) Py_HashPointer(const void *);
PyAPI_FUNC(int) Py_ReprEnter(PyObject *);
PyAPI_FUNC(void) Py_ReprLeave(PyObject *);
PyAPI_FUNC(Py_ssize_t) _Py_HashPointer(const void *);

/* cpyext/osmodule.rs */
PyAPI_FUNC(PyObject *) PyOS_FSPath(PyObject *);

/* cpyext/pyerrors.rs */
PyAPI_FUNC(int) PyErr_BadArgument(void);
PyAPI_FUNC(void) PyErr_BadInternalCall(void);
PyAPI_FUNC(int) PyErr_CheckSignals(void);
PyAPI_FUNC(void) PyErr_Clear(void);
PyAPI_FUNC(int) PyErr_ExceptionMatches(PyObject *);
PyAPI_FUNC(void) PyErr_Fetch(PyObject **, PyObject **, PyObject **);
PyAPI_FUNC(void) PyErr_GetExcInfo(PyObject **, PyObject **, PyObject **);
PyAPI_FUNC(PyObject *) PyErr_GetHandledException(void);
PyAPI_FUNC(PyObject *) PyErr_GetRaisedException(void);
PyAPI_FUNC(int) PyErr_GivenExceptionMatches(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyErr_NoMemory(void);
PyAPI_FUNC(void) PyErr_NormalizeException(PyObject **, PyObject **, PyObject **);
PyAPI_FUNC(PyObject *) PyErr_Occurred(void);
PyAPI_FUNC(void) PyErr_Print(void);
PyAPI_FUNC(void) PyErr_PrintEx(int);
PyAPI_FUNC(void) PyErr_Restore(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(void) PyErr_SetExcInfo(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyErr_SetFromErrno(PyObject *);
PyAPI_FUNC(PyObject *) PyErr_SetFromErrnoWithFilename(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyErr_SetFromErrnoWithFilenameObject(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyErr_SetFromErrnoWithFilenameObjects(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(void) PyErr_SetHandledException(PyObject *);
PyAPI_FUNC(PyObject *) PyErr_SetImportError(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyErr_SetImportErrorSubclass(PyObject *, PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(void) PyErr_SetNone(PyObject *);
PyAPI_FUNC(void) PyErr_SetObject(PyObject *, PyObject *);
PyAPI_FUNC(void) PyErr_SetRaisedException(PyObject *);
PyAPI_FUNC(void) PyErr_SetString(PyObject *, const char *);
PyAPI_FUNC(void) PyErr_WriteUnraisable(PyObject *);
PyAPI_FUNC(int) PyTraceBack_Print(PyObject *, PyObject *);
PyAPI_FUNC(void) _PyErr_BadInternalCall(const char *, int);
PyAPI_FUNC(void) _PyErr_ChainExceptions1(PyObject *);
PyAPI_FUNC(void) _PyPyre_WriteUnraisable(PyObject *, PyObject *);
PyAPI_FUNC(void) _Py_FatalErrorFunc(const char *, const char *);

/* cpyext/pyfile.rs */
PyAPI_FUNC(PyObject *) PyFile_FromFd(int, const char *, const char *, int, const char *, const char *, const char *, int);
PyAPI_FUNC(PyObject *) PyFile_GetLine(PyObject *, int);
PyAPI_FUNC(int) PyFile_WriteObject(PyObject *, PyObject *, int);
PyAPI_FUNC(int) PyFile_WriteString(const char *, PyObject *);

/* cpyext/pymem.rs */
PyAPI_FUNC(void *) PyMem_Calloc(size_t, size_t);
PyAPI_FUNC(void) PyMem_Free(void *);
PyAPI_FUNC(void *) PyMem_Malloc(size_t);
PyAPI_FUNC(void *) PyMem_RawCalloc(size_t, size_t);
PyAPI_FUNC(void) PyMem_RawFree(void *);
PyAPI_FUNC(void *) PyMem_RawMalloc(size_t);
PyAPI_FUNC(void *) PyMem_RawRealloc(void *, size_t);
PyAPI_FUNC(void *) PyMem_Realloc(void *, size_t);

/* cpyext/pyobject.rs */
PyAPI_FUNC(void) Py_DecRef(PyObject *);
PyAPI_FUNC(void) Py_IncRef(PyObject *);
PyAPI_FUNC(Py_ssize_t) _PyPyre_RefCount(PyObject *);
PyAPI_FUNC(void) _Py_DecRef(PyObject *);
PyAPI_FUNC(void) _Py_IncRef(PyObject *);
PyAPI_FUNC(void) _Py_SetRefcnt(PyObject *, Py_ssize_t);

/* cpyext/pystate.rs */
PyAPI_FUNC(void) PyEval_AcquireThread(PyThreadState *);
PyAPI_FUNC(void) PyEval_InitThreads(void);
PyAPI_FUNC(void) PyEval_ReleaseThread(PyThreadState *);
PyAPI_FUNC(void) PyEval_RestoreThread(PyThreadState *);
PyAPI_FUNC(PyThreadState *) PyEval_SaveThread(void);
PyAPI_FUNC(int) PyEval_ThreadsInitialized(void);
PyAPI_FUNC(int) PyGILState_Check(void);
PyAPI_FUNC(PyThreadState *) PyGILState_GetThisThreadState(void);
PyAPI_FUNC(PyInterpreterState *) PyInterpreterState_Get(void);
PyAPI_FUNC(PyObject *) PyInterpreterState_GetDict(PyInterpreterState *);
PyAPI_FUNC(int64_t) PyInterpreterState_GetID(PyInterpreterState *);
PyAPI_FUNC(void) PyThreadState_Clear(PyThreadState *);
PyAPI_FUNC(void) PyThreadState_Delete(PyThreadState *);
PyAPI_FUNC(void) PyThreadState_DeleteCurrent(void);
PyAPI_FUNC(PyThreadState *) PyThreadState_Get(void);
PyAPI_FUNC(PyObject *) PyThreadState_GetDict(void);
PyAPI_FUNC(uint64_t) PyThreadState_GetID(PyThreadState *);
PyAPI_FUNC(PyInterpreterState *) PyThreadState_GetInterpreter(PyThreadState *);
PyAPI_FUNC(PyThreadState *) PyThreadState_GetUnchecked(void);
PyAPI_FUNC(PyThreadState *) PyThreadState_Swap(PyThreadState *);
PyAPI_FUNC(PyObject *) _PyThreadState_GetDict(PyThreadState *);
PyAPI_FUNC(PyThreadState *) _PyThreadState_UncheckedGet(void);

/* cpyext/pystrtod.rs */
PyAPI_FUNC(double) PyOS_string_to_double(const char *, char **, PyObject *);

/* cpyext/pythonrun.rs */
PyAPI_FUNC(int) Py_IsFinalizing(void);
PyAPI_FUNC(int) Py_IsInitialized(void);

/* cpyext/pytime.rs */
PyAPI_FUNC(double) PyTime_AsSecondsDouble(PyTime_t);
PyAPI_FUNC(int) PyTime_Monotonic(PyTime_t *);
PyAPI_FUNC(int) PyTime_MonotonicRaw(PyTime_t *);
PyAPI_FUNC(int) PyTime_PerfCounter(PyTime_t *);
PyAPI_FUNC(int) PyTime_PerfCounterRaw(PyTime_t *);
PyAPI_FUNC(int) PyTime_Time(PyTime_t *);
PyAPI_FUNC(int) PyTime_TimeRaw(PyTime_t *);

/* cpyext/sequence.rs */
PyAPI_FUNC(int) PySequence_Check(PyObject *);
PyAPI_FUNC(PyObject *) PySequence_Concat(PyObject *, PyObject *);
PyAPI_FUNC(int) PySequence_Contains(PyObject *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PySequence_Count(PyObject *, PyObject *);
PyAPI_FUNC(int) PySequence_DelItem(PyObject *, Py_ssize_t);
PyAPI_FUNC(int) PySequence_DelSlice(PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(PyObject *) PySequence_Fast(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PySequence_Fast_GET_ITEM(PyObject *, Py_ssize_t);
PyAPI_FUNC(Py_ssize_t) PySequence_Fast_GET_SIZE(PyObject *);
PyAPI_FUNC(PyObject **) PySequence_Fast_ITEMS(PyObject *);
PyAPI_FUNC(PyObject *) PySequence_GetItem(PyObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PySequence_GetSlice(PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(int) PySequence_In(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PySequence_InPlaceConcat(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PySequence_InPlaceRepeat(PyObject *, Py_ssize_t);
PyAPI_FUNC(Py_ssize_t) PySequence_Index(PyObject *, PyObject *);
PyAPI_FUNC(Py_ssize_t) PySequence_Length(PyObject *);
PyAPI_FUNC(PyObject *) PySequence_List(PyObject *);
PyAPI_FUNC(PyObject *) PySequence_Repeat(PyObject *, Py_ssize_t);
PyAPI_FUNC(int) PySequence_SetItem(PyObject *, Py_ssize_t, PyObject *);
PyAPI_FUNC(int) PySequence_SetSlice(PyObject *, Py_ssize_t, Py_ssize_t, PyObject *);
PyAPI_FUNC(Py_ssize_t) PySequence_Size(PyObject *);
PyAPI_FUNC(PyObject *) PySequence_Tuple(PyObject *);

/* cpyext/setobject.rs */
PyAPI_FUNC(int) PyAnySet_Check(PyObject *);
PyAPI_FUNC(int) PyAnySet_CheckExact(PyObject *);
PyAPI_FUNC(int) PyFrozenSet_Check(PyObject *);
PyAPI_FUNC(int) PyFrozenSet_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyFrozenSet_New(PyObject *);
PyAPI_FUNC(int) PySet_Add(PyObject *, PyObject *);
PyAPI_FUNC(int) PySet_Check(PyObject *);
PyAPI_FUNC(int) PySet_CheckExact(PyObject *);
PyAPI_FUNC(int) PySet_Clear(PyObject *);
PyAPI_FUNC(int) PySet_Contains(PyObject *, PyObject *);
PyAPI_FUNC(int) PySet_Discard(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PySet_New(PyObject *);
PyAPI_FUNC(PyObject *) PySet_Pop(PyObject *);
PyAPI_FUNC(Py_ssize_t) PySet_Size(PyObject *);

/* cpyext/sliceobject.rs */
PyAPI_FUNC(Py_ssize_t) PySlice_AdjustIndices(Py_ssize_t, Py_ssize_t *, Py_ssize_t *, Py_ssize_t);
PyAPI_FUNC(int) PySlice_Check(PyObject *);
PyAPI_FUNC(int) PySlice_GetIndices(PyObject *, Py_ssize_t, Py_ssize_t *, Py_ssize_t *, Py_ssize_t *);
PyAPI_FUNC(int) PySlice_GetIndicesEx(PyObject *, Py_ssize_t, Py_ssize_t *, Py_ssize_t *, Py_ssize_t *, Py_ssize_t *);
PyAPI_FUNC(PyObject *) PySlice_New(PyObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PySlice_Unpack(PyObject *, Py_ssize_t *, Py_ssize_t *, Py_ssize_t *);

/* cpyext/sysmodule.rs */
PyAPI_FUNC(int) PySys_AuditTuple(const char *, PyObject *);
PyAPI_FUNC(PyObject *) PySys_GetObject(const char *);

/* cpyext/tupleobject.rs */
PyAPI_FUNC(int) PyTuple_Check(PyObject *);
PyAPI_FUNC(int) PyTuple_CheckExact(PyObject *);
PyAPI_FUNC(PyObject *) PyTuple_FromArray(PyObject *const *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyTuple_GetItem(PyObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyTuple_GetSlice(PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyTuple_New(Py_ssize_t);
PyAPI_FUNC(int) PyTuple_SetItem(PyObject *, Py_ssize_t, PyObject *);
PyAPI_FUNC(Py_ssize_t) PyTuple_Size(PyObject *);
PyAPI_FUNC(PyObject **) _PyTuple_ITEMS(PyObject *);
PyAPI_FUNC(void) _PyTuple_SET_ITEM(PyObject *, Py_ssize_t, PyObject *);

/* cpyext/typeobject.rs */
PyAPI_FUNC(PyObject *) PyErr_NewException(const char *, PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyErr_NewExceptionWithDoc(const char *, const char *, PyObject *, PyObject *);
PyAPI_FUNC(void) PyObject_Del(void *);
PyAPI_FUNC(void) PyObject_Free(void *);
PyAPI_FUNC(void) PyObject_GC_Del(void *);
PyAPI_FUNC(void *) PyObject_GetItemData(PyObject *);
PyAPI_FUNC(void *) PyObject_GetTypeData(PyObject *, PyTypeObject *);
PyAPI_FUNC(PyObject *) PyObject_Init(PyObject *, PyTypeObject *);
PyAPI_FUNC(int) PyType_Check(PyObject *);
PyAPI_FUNC(unsigned int) PyType_ClearCache(void);
PyAPI_FUNC(int) PyType_Freeze(PyTypeObject *);
PyAPI_FUNC(PyObject *) PyType_FromMetaclass(PyTypeObject *, PyObject *, PyType_Spec *, PyObject *);
PyAPI_FUNC(PyObject *) PyType_FromModuleAndSpec(PyObject *, PyType_Spec *, PyObject *);
PyAPI_FUNC(PyObject *) PyType_FromSlots(PySlot *);
PyAPI_FUNC(PyObject *) PyType_FromSpec(PyType_Spec *);
PyAPI_FUNC(PyObject *) PyType_FromSpecWithBases(PyType_Spec *, PyObject *);
PyAPI_FUNC(PyObject *) PyType_GenericAlloc(PyTypeObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyType_GenericNew(PyTypeObject *, PyObject *, PyObject *);
PyAPI_FUNC(int) PyType_GetBaseByToken(PyTypeObject *, void *, PyTypeObject **);
PyAPI_FUNC(PyObject *) PyType_GetDict(PyTypeObject *);
PyAPI_FUNC(unsigned long) PyType_GetFlags(PyTypeObject *);
PyAPI_FUNC(PyObject *) PyType_GetFullyQualifiedName(PyTypeObject *);
PyAPI_FUNC(PyObject *) PyType_GetModule(PyTypeObject *);
PyAPI_FUNC(PyObject *) PyType_GetModuleByDef(PyTypeObject *, PyModuleDef *);
PyAPI_FUNC(PyObject *) PyType_GetModuleName(PyTypeObject *);
PyAPI_FUNC(void *) PyType_GetModuleState(PyTypeObject *);
PyAPI_FUNC(PyObject *) PyType_GetName(PyTypeObject *);
PyAPI_FUNC(PyObject *) PyType_GetQualName(PyTypeObject *);
PyAPI_FUNC(void *) PyType_GetSlot(PyTypeObject *, int);
PyAPI_FUNC(Py_ssize_t) PyType_GetTypeDataSize(PyTypeObject *);
PyAPI_FUNC(int) PyType_IsSubtype(PyTypeObject *, PyTypeObject *);
PyAPI_FUNC(void) PyType_Modified(PyTypeObject *);
PyAPI_FUNC(int) PyType_Ready(PyTypeObject *);
PyAPI_FUNC(const char *) _PyType_Name(PyTypeObject *);

/* cpyext/unicodeobject.rs */
PyAPI_FUNC(void) PyUnicode_Append(PyObject **, PyObject *);
PyAPI_FUNC(void) PyUnicode_AppendAndDel(PyObject **, PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_AsASCIIString(PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_AsEncodedString(PyObject *, const char *, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_AsLatin1String(PyObject *);
PyAPI_FUNC(Py_UCS4 *) PyUnicode_AsUCS4(PyObject *, Py_UCS4 *, Py_ssize_t, int);
PyAPI_FUNC(Py_UCS4 *) PyUnicode_AsUCS4Copy(PyObject *);
PyAPI_FUNC(const char *) PyUnicode_AsUTF8(PyObject *);
PyAPI_FUNC(const char *) PyUnicode_AsUTF8AndSize(PyObject *, Py_ssize_t *);
PyAPI_FUNC(PyObject *) PyUnicode_AsUTF8String(PyObject *);
PyAPI_FUNC(Py_ssize_t) PyUnicode_AsWideChar(PyObject *, wchar_t *, Py_ssize_t);
PyAPI_FUNC(wchar_t *) PyUnicode_AsWideCharString(PyObject *, Py_ssize_t *);
PyAPI_FUNC(int) PyUnicode_Check(PyObject *);
PyAPI_FUNC(int) PyUnicode_CheckExact(PyObject *);
PyAPI_FUNC(int) PyUnicode_Compare(PyObject *, PyObject *);
PyAPI_FUNC(int) PyUnicode_CompareWithASCIIString(PyObject *, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_Concat(PyObject *, PyObject *);
PyAPI_FUNC(int) PyUnicode_Contains(PyObject *, PyObject *);
PyAPI_FUNC(void *) PyUnicode_DATA(PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_Decode(const char *, Py_ssize_t, const char *, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeASCII(const char *, Py_ssize_t, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeFSDefault(const char *);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeFSDefaultAndSize(const char *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeLatin1(const char *, Py_ssize_t, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeLocale(const char *, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeLocaleAndSize(const char *, Py_ssize_t, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_DecodeUTF8(const char *, Py_ssize_t, const char *);
PyAPI_FUNC(PyObject *) PyUnicode_EncodeFSDefault(PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_EncodeLocale(PyObject *, const char *);
PyAPI_FUNC(int) PyUnicode_Equal(PyObject *, PyObject *);
PyAPI_FUNC(int) PyUnicode_EqualToUTF8(PyObject *, const char *);
PyAPI_FUNC(int) PyUnicode_EqualToUTF8AndSize(PyObject *, const char *, Py_ssize_t);
PyAPI_FUNC(int) PyUnicode_FSConverter(PyObject *, void *);
PyAPI_FUNC(int) PyUnicode_FSDecoder(PyObject *, void *);
PyAPI_FUNC(Py_ssize_t) PyUnicode_FindChar(PyObject *, Py_UCS4, Py_ssize_t, Py_ssize_t, int);
PyAPI_FUNC(PyObject *) PyUnicode_Format(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_FromKindAndData(int, const void *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyUnicode_FromObject(PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_FromOrdinal(int);
PyAPI_FUNC(PyObject *) PyUnicode_FromString(const char *);
PyAPI_FUNC(PyObject *) PyUnicode_FromStringAndSize(const char *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyUnicode_FromWideChar(const wchar_t *, Py_ssize_t);
PyAPI_FUNC(Py_ssize_t) PyUnicode_GetLength(PyObject *);
PyAPI_FUNC(unsigned int) PyUnicode_IS_ASCII(PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_InternFromString(const char *);
PyAPI_FUNC(void) PyUnicode_InternInPlace(PyObject **);
PyAPI_FUNC(PyObject *) PyUnicode_Join(PyObject *, PyObject *);
PyAPI_FUNC(int) PyUnicode_KIND(PyObject *);
PyAPI_FUNC(unsigned int) PyUnicode_MAX_CHAR_VALUE(PyObject *);
PyAPI_FUNC(PyObject *) PyUnicode_New(Py_ssize_t, Py_UCS4);
PyAPI_FUNC(Py_UCS4) PyUnicode_ReadChar(PyObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyUnicode_RichCompare(PyObject *, PyObject *, int);
PyAPI_FUNC(PyObject *) PyUnicode_Split(PyObject *, PyObject *, Py_ssize_t);
PyAPI_FUNC(PyObject *) PyUnicode_Substring(PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(int) PyUnicode_WriteChar(PyObject *, Py_ssize_t, Py_UCS4);

/* cpyext/unicodewriter.rs */
PyAPI_FUNC(PyUnicodeWriter *) PyUnicodeWriter_Create(Py_ssize_t);
PyAPI_FUNC(void) PyUnicodeWriter_Discard(PyUnicodeWriter *);
PyAPI_FUNC(PyObject *) PyUnicodeWriter_Finish(PyUnicodeWriter *);
PyAPI_FUNC(int) PyUnicodeWriter_WriteASCII(PyUnicodeWriter *, const char *, Py_ssize_t);
PyAPI_FUNC(int) PyUnicodeWriter_WriteChar(PyUnicodeWriter *, Py_UCS4);
PyAPI_FUNC(int) PyUnicodeWriter_WriteRepr(PyUnicodeWriter *, PyObject *);
PyAPI_FUNC(int) PyUnicodeWriter_WriteStr(PyUnicodeWriter *, PyObject *);
PyAPI_FUNC(int) PyUnicodeWriter_WriteSubstring(PyUnicodeWriter *, PyObject *, Py_ssize_t, Py_ssize_t);
PyAPI_FUNC(int) PyUnicodeWriter_WriteUCS4(PyUnicodeWriter *, Py_UCS4 *, Py_ssize_t);
PyAPI_FUNC(int) PyUnicodeWriter_WriteUTF8(PyUnicodeWriter *, const char *, Py_ssize_t);
PyAPI_FUNC(int) PyUnicodeWriter_WriteWideChar(PyUnicodeWriter *, const wchar_t *, Py_ssize_t);

/* cpyext/warnings.rs */
PyAPI_FUNC(int) PyErr_WarnEx(PyObject *, const char *, Py_ssize_t);
PyAPI_FUNC(int) PyErr_WarnExplicit(PyObject *, const char *, const char *, int, const char *, PyObject *);
PyAPI_FUNC(int) PyErr_WarnExplicitObject(PyObject *, PyObject *, PyObject *, int, PyObject *, PyObject *);
PyAPI_FUNC(int) _PyPyre_WarnExplicitMessage(PyObject *, PyObject *, const char *, int, const char *, PyObject *);
PyAPI_FUNC(int) _PyPyre_WarnUnicode(PyObject *, PyObject *, PyObject *, Py_ssize_t);

/* cpyext/weakrefobject.rs */
PyAPI_FUNC(void) PyObject_ClearWeakRefs(PyObject *);
PyAPI_FUNC(int) PyWeakref_Check(PyObject *);
PyAPI_FUNC(int) PyWeakref_CheckProxy(PyObject *);
PyAPI_FUNC(int) PyWeakref_CheckRef(PyObject *);
PyAPI_FUNC(int) PyWeakref_CheckRefExact(PyObject *);
PyAPI_FUNC(PyObject *) PyWeakref_GetObject(PyObject *);
PyAPI_FUNC(int) PyWeakref_GetRef(PyObject *, PyObject **);
PyAPI_FUNC(int) PyWeakref_IsDead(PyObject *);
PyAPI_FUNC(PyObject *) PyWeakref_NewProxy(PyObject *, PyObject *);
PyAPI_FUNC(PyObject *) PyWeakref_NewRef(PyObject *, PyObject *);

#ifdef __cplusplus
}
#endif

#endif /* !PYRE_DECL_H */
