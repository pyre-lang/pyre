"""App-level _typing — the type-parameter runtime objects typing.py imports.

These mirror the objects in Objects/typevarobject.c: TypeVar, ParamSpec,
TypeVarTuple, ParamSpecArgs/Kwargs, TypeAliasType, Generic, plus the
NoDefault sentinel.  The heavy substitution and
class-getitem logic lives in typing.py; these objects delegate to its
module-level helpers (_typevar_subst, _paramspec_subst, _generic_class_getitem,
...), exactly as the C objects call back into the typing module.
"""

# `_typing.c` names each of these types `typing.X`, because `typing` is the
# public owner their repr, their pickle and their substitution semantics all
# report.  This file backs `_typing`, so the name the install binds is that
# one; rebind it to the owner the classes below must claim.
__name__ = "typing"

import sys
from types import GenericAlias, UnionType as Union


def _caller_module():
    # `typevarobject.c caller()` reads `__name__` out of the globals of the
    # frame that is running, and the constructor calling it is native, so the
    # frame it finds is the user's.  The frames of this module are not the
    # program's either -- `sys._getframe` counts along a walk that steps over
    # them -- so depth 0 names that same frame however deep the helper sits.
    try:
        return sys._getframe(0).f_globals.get('__name__')
    except (AttributeError, ValueError):
        return None


class _PickleUsingNameMixin:
    """PyPy's name-based reducer for runtime typing objects."""

    __slots__ = ()

    def __reduce__(self):
        return self.__name__


def _immutable_type_error(cls, name):
    return TypeError(
        f"cannot set {name!r} attribute of immutable type "
        f"'{cls.__module__}.{cls.__name__}'"
    )


class _ImmutableTypeMeta(type):
    """``Py_TPFLAGS_IMMUTABLETYPE`` for app-level typing types.

    The guard is the metaclass hook, so it answers ``setattr``/``delattr`` but
    not an explicit ``type.__setattr__(TypeVar, ...)``, which reaches the base
    implementation and still mutates the class.  Closing that would need an
    immutability bit separate from heaptype, and ``W_TypeObject`` declares
    ``flag_heaptype`` in ``_immutable_fields_`` unqualified -- a JIT hint
    governs the value, so the split stays PyPy's to make.
    """

    def __new__(mcls, name, bases, namespace):
        for base in bases:
            if isinstance(base, _ImmutableTypeMeta):
                raise TypeError(
                    f"type '{base.__module__}.{base.__name__}' "
                    "is not an acceptable base type"
                )
        return super().__new__(mcls, name, bases, namespace)

    def __setattr__(cls, name, value):
        raise _immutable_type_error(cls, name)

    def __delattr__(cls, name):
        # `type_setattro` uses the same immutable-type error for deletion.
        raise _immutable_type_error(cls, name)


def _readonly_member(slot):
    """Expose a private slot as a `Py_READONLY` struct member.

    A slot named for the public attribute installs a writable member
    descriptor, which `object.__setattr__` reaches past the `__setattr__`
    guard below.  A property is a data descriptor, so it answers every path,
    and `member_set` phrases both the write and the delete the same way.
    """

    def get(self):
        return object.__getattribute__(self, slot)

    def refuse(self, *_value):
        raise AttributeError("readonly attribute")

    return property(get, refuse, refuse)


class _Immutable:
    """PyPy's per-instance readonly-field mixin.

    CPython 3.14 keeps a managed attribute dictionary on TypeVar, ParamSpec,
    and TypeVarTuple, but does not expose it as ``__dict__``.  Their native
    struct members and getsets stay read-only while unrelated user attributes
    remain writable.  ``None`` means the instance has no user dictionary.

    The readonly guard is ``__setattr__``/``__getattribute__``, so
    ``object.__setattr__(t, '__name__', ...)`` and
    ``object.__getattribute__(t, '__dict__')`` still reach the real instance
    dict.  Native per-field storage would close both, but it moves the storage
    owner and clears ``hasdict``, which ``W_TypeObject`` also declares in
    ``_immutable_fields_``.
    """

    __slots__ = ()

    _readonly_attrs = None
    _readonly_members = frozenset()
    # Slots that carry a member's storage under a private name because the
    # public one is a read-only descriptor.  The native types have no such
    # attribute, so `dir()` must not report it either.
    _hidden_slots = frozenset()

    def __getattribute__(self, name):
        if name == "__dict__":
            qualname = f"{type(self).__module__}.{type(self).__name__}"
            raise AttributeError(
                f"{qualname!r} object has no attribute '__dict__'"
            )
        return object.__getattribute__(self, name)

    def __dir__(self):
        hidden = {"__dict__", "__weakref__"} | set(type(self)._hidden_slots)
        return [name for name in object.__dir__(self) if name not in hidden]

    def __setattr__(self, name, value):
        readonly = type(self)._readonly_attrs
        if name in type(self)._readonly_members:
            raise AttributeError("readonly attribute")
        if readonly is not None and name in readonly:
            qualname = f"{type(self).__module__}.{type(self).__name__}"
            raise AttributeError(
                f"attribute {name!r} of {qualname!r} objects is not writable"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        readonly = type(self)._readonly_attrs
        if name in type(self)._readonly_members:
            raise AttributeError("readonly attribute")
        if readonly is not None and name in readonly:
            qualname = f"{type(self).__module__}.{type(self).__name__}"
            raise AttributeError(
                f"attribute {name!r} of {qualname!r} objects is not writable"
            )
        object.__delattr__(self, name)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _evaluate_typeparam(thunk):
    # Call a PEP 695 bound / constraints thunk emitted by the compiler. A 3.14
    # thunk takes an annotation `format` argument (annotationlib.Format.VALUE
    # == 1); accept a zero-argument thunk as well.
    code = getattr(thunk, '__code__', None)
    if code is not None and code.co_argcount >= 1:
        return thunk(1)
    return thunk()


class _NoDefaultMeta(type):
    def __new__(mcls, name, bases, namespace):
        if any(isinstance(base, _NoDefaultMeta) for base in bases):
            raise TypeError("type 'typing.NoDefault' is not an acceptable base type")
        return super().__new__(mcls, name, bases, namespace)

    def __setattr__(cls, name, value):
        raise TypeError("cannot set attributes of immutable type 'typing.NoDefault'")


_no_default_instance = None


class _NoDefaultType(metaclass=_NoDefaultMeta):
    """Type of the typing.NoDefault sentinel."""

    __slots__ = ()

    def __new__(cls):
        global _no_default_instance
        if _no_default_instance is None:
            _no_default_instance = super().__new__(cls)
        return _no_default_instance

    def __repr__(self):
        return "typing.NoDefault"

    def __reduce__(self):
        return "NoDefault"


NoDefault = _NoDefaultType()


_MISSING = object()


def _typing_type_repr(value):
    """CPython 3.14 `_Py_typing_type_repr`, for constant evaluators."""
    if value is Ellipsis:
        return "..."
    if value is type(None):
        return "None"
    if hasattr(value, "__origin__") and hasattr(value, "__args__"):
        return repr(value)
    qualname = getattr(value, "__qualname__", None)
    module = getattr(value, "__module__", None)
    if qualname is None or module is None:
        return repr(value)
    if module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def _index(value) -> int:
    # `operator.index`, written out because this module loads before the
    # pure-Python `operator` is importable.  An evaluator's format argument is
    # a `Format` member, so it reaches here through the integer index
    # protocol: a float or a string is a TypeError rather than a format that
    # silently compares unequal to every member.
    if isinstance(value, int):
        result = value
    else:
        try:
            index = type(value).__index__
        except AttributeError:
            raise TypeError(
                f"{type(value).__name__!r} object cannot be interpreted as an integer"
            ) from None
        result = index(value)
        if not isinstance(result, int):
            raise TypeError(f"__index__ returned non-int (type {type(result).__name__})")
    # `int(result)` would route a subclass through its own `__int__`, which may
    # answer something other than the value the subclass stores. Narrowing to
    # an exact `int` is what `PyNumber_Index` does with `_PyLong_Copy`.
    return int.__index__(result)


def _immutable_const_evaluator_error(name):
    return TypeError(
        f"cannot set '{name}' attribute of immutable type "
        "'_typing._ConstEvaluator'"
    )


class _ConstEvaluatorMeta(type):
    def __setattr__(cls, name, value):
        raise _immutable_const_evaluator_error(name)

    def __delattr__(cls, name):
        # `type_setattro` handles both writes, so an immutable type refuses a
        # deletion with the wording it uses for an assignment. Without this,
        # `del type(evaluator).__call__` removes call support from every
        # constant evaluator in the process.
        raise _immutable_const_evaluator_error(name)


class _ConstEvaluator(metaclass=_ConstEvaluatorMeta):
    """CPython 3.14 `constevaluatorobject`."""

    __slots__ = ("_value",)

    def __new__(cls):
        raise TypeError("cannot create '_typing._ConstEvaluator' instances")

    def __call__(self, format, /):
        # The parameter keeps CPython's name; the converted value gets its own
        # local so the builtin stays reachable in this scope.
        format_value = _index(format)
        if format_value == 4:  # annotationlib.Format.STRING
            if isinstance(self._value, tuple):
                return "(" + ", ".join(map(_typing_type_repr, self._value)) + ")"
            return _typing_type_repr(self._value)
        return self._value

    def __repr__(self):
        return f"<constevaluator {self._value!r}>"


def _const_evaluator(value):
    evaluator = object.__new__(_ConstEvaluator)
    evaluator._value = value
    return evaluator


def _variance_prefix(infer_variance, covariant, contravariant):
    if infer_variance:
        return ''
    if covariant:
        return '+'
    if contravariant:
        return '-'
    return '~'


class TypeVar(
    _Immutable, _PickleUsingNameMixin, metaclass=_ImmutableTypeMeta
):
    """Type variable — PEP 484 / PEP 695."""

    _readonly_members = frozenset((
        "__name__", "__covariant__", "__contravariant__",
        "__infer_variance__",
    ))
    _readonly_attrs = frozenset((
        "__bound__", "__constraints__", "__default__", "evaluate_bound",
        "evaluate_constraints", "evaluate_default", "_bound",
        "_constraints", "_default_value", "_evaluate_bound",
        "_evaluate_constraints", "_evaluate_default",
    ))

    def __init__(self, name, *constraints, bound=None, default=NoDefault,
                 covariant=False, contravariant=False, infer_variance=False):
        if not isinstance(name, str):
            raise TypeError(
                f"typevar() argument 'name' must be str, not "
                f"{type(name).__name__}"
            )
        object.__setattr__(self, "__name__", name)
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        if infer_variance and (covariant or contravariant):
            raise ValueError("Variance cannot be specified with infer_variance.")
        object.__setattr__(self, "__covariant__", bool(covariant))
        object.__setattr__(self, "__contravariant__", bool(contravariant))
        object.__setattr__(self, "__infer_variance__", bool(infer_variance))
        object.__setattr__(self, "_default_value", default)
        object.__setattr__(self, "_evaluate_default", None)
        if constraints and bound is not None:
            raise TypeError("Constraints cannot be combined with bound=...")
        if len(constraints) == 1:
            raise TypeError("A single constraint is not allowed")
        import typing
        constraints = tuple(
            typing._type_check(
                constraint,
                f"TypeVar(name, constraint, ...). Constraints must be types. Got {constraint!r}.",
            )
            for constraint in constraints
        )
        object.__setattr__(self, "_constraints", constraints)
        object.__setattr__(self, "_evaluate_constraints", None)
        bound = (
            None
            if bound is None
            else typing._type_check(bound, "Bound must be a type.")
        )
        object.__setattr__(self, "_bound", bound)
        object.__setattr__(self, "_evaluate_bound", None)
        object.__setattr__(self, "__module__", _caller_module())

    @classmethod
    def _make(cls, name, *, evaluate_bound=None, evaluate_constraints=None):
        # Lazy construction for the TYPEVAR_WITH_BOUND and
        # TYPEVAR_WITH_CONSTRAINTS intrinsics. The bound / constraints arrive as
        # thunks the compiler defers so they may reference names bound later in
        # the enclosing scope; they are evaluated on first `__bound__` /
        # `__constraints__` access and cached (Objects/typevarobject.c).
        self = cls.__new__(cls)
        object.__setattr__(self, "__name__", name)
        object.__setattr__(self, "__covariant__", False)
        object.__setattr__(self, "__contravariant__", False)
        # CPython 3.14 `_Py_make_typevar`: type parameters created by the
        # compiler infer variance, unlike an ordinary `TypeVar(...)` call.
        object.__setattr__(self, "__infer_variance__", True)
        # `_Py_make_typevar` leaves `default_value` NULL, which `_MISSING`
        # stands for.  A `TypeVar(...)` call instead stores the `NoDefault`
        # sentinel its signature defaults to, and the two states differ:
        # `evaluate_default` answers `None` for the first and a constant
        # evaluator over `NoDefault` for the second.
        object.__setattr__(self, "_default_value", _MISSING)
        object.__setattr__(self, "_evaluate_default", None)
        object.__setattr__(
            self, "_constraints",
            _MISSING if evaluate_constraints is not None else (),
        )
        object.__setattr__(self, "_evaluate_constraints", evaluate_constraints)
        object.__setattr__(
            self, "_bound", _MISSING if evaluate_bound is not None else None
        )
        object.__setattr__(self, "_evaluate_bound", evaluate_bound)
        # `_Py_make_typevar` passes a NULL module, so the compiler-created
        # parameter never gets an instance `__module__` and reports the class
        # attribute instead.
        return self

    @property
    def __bound__(self):
        if self._bound is _MISSING:
            object.__setattr__(
                self, "_bound", _evaluate_typeparam(self._evaluate_bound)
            )
        return self._bound

    @property
    def __constraints__(self):
        if self._constraints is _MISSING:
            object.__setattr__(
                self, "_constraints",
                tuple(_evaluate_typeparam(self._evaluate_constraints)),
            )
        return self._constraints

    @property
    def __default__(self):
        if self._default_value is _MISSING:
            if self._evaluate_default is None:
                return NoDefault
            object.__setattr__(
                self, "_default_value",
                _evaluate_typeparam(self._evaluate_default),
            )
        return self._default_value

    @property
    def evaluate_bound(self):
        if self._evaluate_bound is not None:
            return self._evaluate_bound
        if self._bound is not None:
            return _const_evaluator(self._bound)
        return None

    @property
    def evaluate_constraints(self):
        if self._evaluate_constraints is not None:
            return self._evaluate_constraints
        if self._constraints:
            return _const_evaluator(self._constraints)
        return None

    @property
    def evaluate_default(self):
        if self._evaluate_default is not None:
            return self._evaluate_default
        if self._default_value is _MISSING:
            return None
        return _const_evaluator(self._default_value)

    def __typing_subst__(self, arg):
        import typing
        return typing._typevar_subst(self, arg)

    def __typing_prepare_subst__(self, alias, args):
        params = alias.__parameters__
        try:
            index = list(params).index(self)
        except ValueError:
            return args
        if len(args) == index and self.has_default():
            args = list(args)
            args.append(self.__default__)
            args = tuple(args)
        return args

    def has_default(self):
        return self._evaluate_default is not None or (
            self._default_value is not _MISSING
            and self._default_value is not NoDefault
        )

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of TypeVar")

    def __or__(self, other):
        import typing
        return Union[self, typing._type_convert(other)]

    def __ror__(self, other):
        import typing
        return Union[typing._type_convert(other), self]

    def __repr__(self):
        return _variance_prefix(self.__infer_variance__, self.__covariant__,
                                self.__contravariant__) + self.__name__

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.TypeVar' is not an acceptable base type")


class ParamSpec(
    _Immutable, _PickleUsingNameMixin, metaclass=_ImmutableTypeMeta
):
    """Parameter specification variable — PEP 612."""

    _readonly_members = frozenset((
        "__name__", "__bound__", "__covariant__", "__contravariant__",
        "__infer_variance__",
    ))
    _readonly_attrs = frozenset((
        "args", "kwargs", "__default__", "evaluate_default",
        "_default_value", "_evaluate_default",
    ))

    def __init__(self, name, *, bound=None, default=NoDefault,
                 covariant=False, contravariant=False, infer_variance=False):
        if not isinstance(name, str):
            raise TypeError(
                f"paramspec() argument 'name' must be str, not "
                f"{type(name).__name__}"
            )
        object.__setattr__(self, "__name__", name)
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        if infer_variance and (covariant or contravariant):
            raise ValueError("Variance cannot be specified with infer_variance.")
        object.__setattr__(self, "__covariant__", bool(covariant))
        object.__setattr__(self, "__contravariant__", bool(contravariant))
        object.__setattr__(self, "__infer_variance__", bool(infer_variance))
        object.__setattr__(self, "_default_value", default)
        object.__setattr__(self, "_evaluate_default", None)
        # `paramspec_new_impl` stores `bound` through `paramspec_alloc` /
        # `Py_XNewRef` with no type check, so an omitted bound stays `None`.
        object.__setattr__(self, "__bound__", bound)
        object.__setattr__(self, "__module__", _caller_module())

    @classmethod
    def _make(cls, name):
        # `_Py_make_paramspec`: the compiler-created parameter infers variance
        # and leaves `default_value` NULL, which `_MISSING` stands for.
        self = cls.__new__(cls)
        object.__setattr__(self, "__name__", name)
        object.__setattr__(self, "__covariant__", False)
        object.__setattr__(self, "__contravariant__", False)
        object.__setattr__(self, "__infer_variance__", True)
        object.__setattr__(self, "_default_value", _MISSING)
        object.__setattr__(self, "_evaluate_default", None)
        object.__setattr__(self, "__bound__", None)
        # `paramspec_alloc` is handed a NULL module here, so `__module__` stays
        # the class attribute.
        return self

    @property
    def args(self):
        return ParamSpecArgs(self)

    @property
    def kwargs(self):
        return ParamSpecKwargs(self)

    def __typing_subst__(self, arg):
        import typing
        return typing._paramspec_subst(self, arg)

    def __typing_prepare_subst__(self, alias, args):
        import typing
        return typing._paramspec_prepare_subst(self, alias, args)

    def has_default(self):
        return self._evaluate_default is not None or (
            self._default_value is not _MISSING
            and self._default_value is not NoDefault
        )

    @property
    def __default__(self):
        if self._default_value is _MISSING:
            if self._evaluate_default is None:
                return NoDefault
            object.__setattr__(
                self, "_default_value",
                _evaluate_typeparam(self._evaluate_default),
            )
        return self._default_value

    @property
    def evaluate_default(self):
        if self._evaluate_default is not None:
            return self._evaluate_default
        if self._default_value is _MISSING:
            return None
        return _const_evaluator(self._default_value)

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpec")

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.ParamSpec' is not an acceptable base type")

    def __repr__(self):
        return _variance_prefix(self.__infer_variance__, self.__covariant__,
                                self.__contravariant__) + self.__name__


class ParamSpecArgs(_Immutable, metaclass=_ImmutableTypeMeta):
    """The args of a ParamSpec, e.g. P.args."""

    # `paramspecargs` stores the origin in a `Py_READONLY` member, so a slot
    # named `__origin__` would be writable through `object.__setattr__` where
    # the native view is not.  The storage takes a private name and the public
    # one is a read-only property, which is a data descriptor and so refuses
    # the write on both paths.
    __slots__ = ("_origin", "__weakref__")
    _readonly_members = frozenset(("__origin__",))
    _hidden_slots = frozenset(("_origin",))

    def __init__(self, origin):
        object.__setattr__(self, "_origin", origin)

    __origin__ = _readonly_member("_origin")

    def __repr__(self):
        if type(self.__origin__) is ParamSpec:
            return f"{self.__origin__.__name__}.args"
        return f"{self.__origin__!r}.args"

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self.__origin__ == other.__origin__

    # `P.args` builds a fresh object per access while `__eq__` compares by
    # origin, so an identity hash would give equal objects different hashes.
    # The native `paramspecargs` supplies `tp_richcompare` and no `tp_hash`,
    # which is the unhashable that leaving `__hash__` alone reproduces here.

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpecArgs")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.ParamSpecArgs' is not an acceptable base type")


class ParamSpecKwargs(_Immutable, metaclass=_ImmutableTypeMeta):
    """The kwargs of a ParamSpec, e.g. P.kwargs."""

    # Read-only through a private slot, for the reason given on
    # `ParamSpecArgs`.
    __slots__ = ("_origin", "__weakref__")
    _readonly_members = frozenset(("__origin__",))
    _hidden_slots = frozenset(("_origin",))

    def __init__(self, origin):
        object.__setattr__(self, "_origin", origin)

    __origin__ = _readonly_member("_origin")

    def __repr__(self):
        if type(self.__origin__) is ParamSpec:
            return f"{self.__origin__.__name__}.kwargs"
        return f"{self.__origin__!r}.kwargs"

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self.__origin__ == other.__origin__

    # Unhashable for the reason given on `ParamSpecArgs`.

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of ParamSpecKwargs")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.ParamSpecKwargs' is not an acceptable base type")


class TypeVarTuple(
    _Immutable, _PickleUsingNameMixin, metaclass=_ImmutableTypeMeta
):
    """Type variable tuple — PEP 646."""

    _readonly_members = frozenset((
        "__name__", "__bound__", "__covariant__", "__contravariant__",
        "__infer_variance__",
    ))
    _readonly_attrs = frozenset((
        "__default__", "evaluate_default", "_default_value",
        "_evaluate_default",
    ))

    def __init__(self, name, *, bound=None, covariant=False,
                 contravariant=False, infer_variance=False, default=NoDefault):
        if not isinstance(name, str):
            raise TypeError(
                f"typevartuple() argument 'name' must be str, not "
                f"{type(name).__name__}"
            )
        object.__setattr__(self, "__name__", name)
        if covariant and contravariant:
            raise ValueError("Bivariant types are not supported.")
        if infer_variance and (covariant or contravariant):
            raise ValueError("Variance cannot be specified with infer_variance.")
        object.__setattr__(self, "__covariant__", bool(covariant))
        object.__setattr__(self, "__contravariant__", bool(contravariant))
        object.__setattr__(self, "__infer_variance__", bool(infer_variance))
        # `typevartuple_impl` stores `bound` through `typevartuple_alloc` /
        # `Py_XNewRef` with no type check, so an omitted bound stays `None`.
        object.__setattr__(self, "__bound__", bound)
        object.__setattr__(self, "_default_value", default)
        object.__setattr__(self, "_evaluate_default", None)
        object.__setattr__(self, "__module__", _caller_module())

    @classmethod
    def _make(cls, name):
        # `_Py_make_typevartuple` leaves `default_value` NULL, which `_MISSING`
        # stands for, and infers variance.
        self = cls.__new__(cls)
        object.__setattr__(self, "__name__", name)
        object.__setattr__(self, "__covariant__", False)
        object.__setattr__(self, "__contravariant__", False)
        object.__setattr__(self, "__infer_variance__", True)
        object.__setattr__(self, "__bound__", None)
        object.__setattr__(self, "_default_value", _MISSING)
        object.__setattr__(self, "_evaluate_default", None)
        # `typevartuple_alloc` takes no module here, so `__module__` stays the
        # class attribute.
        return self

    def __iter__(self):
        import typing
        yield typing.Unpack[self]

    def __typing_subst__(self, arg):
        raise TypeError("Substitution of bare TypeVarTuple is not supported")

    def __typing_prepare_subst__(self, alias, args):
        import typing
        return typing._typevartuple_prepare_subst(self, alias, args)

    def has_default(self):
        return self._evaluate_default is not None or (
            self._default_value is not _MISSING
            and self._default_value is not NoDefault
        )

    @property
    def __default__(self):
        if self._default_value is _MISSING:
            if self._evaluate_default is None:
                return NoDefault
            object.__setattr__(
                self, "_default_value",
                _evaluate_typeparam(self._evaluate_default),
            )
        return self._default_value

    @property
    def evaluate_default(self):
        if self._evaluate_default is not None:
            return self._evaluate_default
        if self._default_value is _MISSING:
            return None
        return _const_evaluator(self._default_value)

    def __mro_entries__(self, bases):
        raise TypeError("Cannot subclass an instance of TypeVarTuple")

    def __repr__(self):
        return _variance_prefix(self.__infer_variance__, self.__covariant__,
                                self.__contravariant__) + self.__name__

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'typing.TypeVarTuple' is not an acceptable base type")


class TypeAliasType(_PickleUsingNameMixin, metaclass=_ImmutableTypeMeta):
    """A PEP 695 ``type X = ...`` alias."""

    __slots__ = ("_name", "_type_params", "_value", "_evaluate_value", "_module")

    def __init__(self, name, value, *, type_params=()):
        if not isinstance(name, str):
            raise TypeError(
                f"typealias() argument 'name' must be str, not "
                f"{type(name).__name__}"
            )
        if not isinstance(type_params, tuple):
            raise TypeError("type_params must be a tuple")
        self._check_type_params(type_params)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_type_params", type_params)
        object.__setattr__(self, "_value", value)
        object.__setattr__(self, "_evaluate_value", None)
        object.__setattr__(self, "_module", _caller_module())

    @staticmethod
    def _check_type_params(type_params):
        default_seen = False
        for param in type_params:
            if not (
                type(param) is TypeVar
                or type(param) is ParamSpec
                or type(param) is TypeVarTuple
            ):
                raise TypeError(f"Expected a type param, got {param!r}")
            default = param.__default__
            if default is NoDefault:
                if default_seen:
                    raise TypeError(
                        f"non-default type parameter '{param!r}' "
                        "follows default type parameter"
                    )
            else:
                default_seen = True

    @classmethod
    def _from_evaluator(cls, name, type_params, evaluate_value):
        # PyPy's `_make_typealiastype` allocates first and installs the lazy
        # evaluator with `object.__setattr__`.  Keep that shape so the public
        # constructor does not grow a CPython-incompatible private keyword.
        self = object.__new__(cls)
        # CPython `_Py_make_typealias` trusts the compiler-provided tuple and
        # deliberately skips `typealias_check_type_params`: checking here
        # would force lazy defaults before the alias is even constructed.
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_type_params", type_params)
        object.__setattr__(self, "_value", _MISSING)
        object.__setattr__(self, "_evaluate_value", evaluate_value)
        # `_Py_make_typealias` stores no module.  The `__module__` getter
        # derives it from the evaluator function when first observed.
        object.__setattr__(self, "_module", _MISSING)
        return self

    def __getattribute__(self, name):
        if name == "__module__":
            module = object.__getattribute__(self, "_module")
            if module is not _MISSING:
                return module
            evaluate_value = object.__getattribute__(self, "_evaluate_value")
            return getattr(evaluate_value, "__module__", None)
        return object.__getattribute__(self, name)

    def __setattr__(self, name, value):
        if name == "__name__":
            raise AttributeError("readonly attribute")
        if name in {
            "__module__", "__parameters__", "__type_params__", "__value__",
            "evaluate_value",
        }:
            raise AttributeError(
                f"attribute {name!r} of 'typing.TypeAliasType' objects "
                "is not writable"
            )
        raise AttributeError(
            f"'typing.TypeAliasType' object has no attribute {name!r} "
            "and no __dict__ for setting new attributes"
        )

    def __delattr__(self, name):
        # The native object uses the same read-only/no-dict paths for writes
        # and deletions.
        self.__setattr__(name, None)

    @property
    def __name__(self):
        return self._name

    @property
    def __type_params__(self):
        return self._type_params

    @property
    def __value__(self):
        if self._value is _MISSING:
            object.__setattr__(
                self, "_value", _evaluate_typeparam(self._evaluate_value)
            )
        return self._value

    @property
    def evaluate_value(self):
        if self._evaluate_value is not None:
            return self._evaluate_value
        return _const_evaluator(self._value)

    @property
    def __parameters__(self):
        if not self._type_params:
            return ()
        if not any(type(param) is TypeVarTuple for param in self._type_params):
            # CPython's `unpack_typevartuples` returns the original tuple when
            # there is nothing to unpack.
            return self._type_params
        result = []
        for param in self._type_params:
            if type(param) is TypeVarTuple:
                # PyPy spells this `result.extend(param)`: TypeVarTuple's
                # one-item iterator produces typing.Unpack[param].
                result.extend(param)
            else:
                result.append(param)
        return tuple(result)

    def __iter__(self):
        import typing
        yield typing.Unpack[self]

    def __getitem__(self, args):
        if not self.__type_params__:
            raise TypeError("Only generic type aliases are subscriptable")
        if not isinstance(args, tuple):
            args = (args,)
        return GenericAlias(self, args)

    def __or__(self, other):
        return Union[self, other]

    def __ror__(self, other):
        return Union[other, self]

    def __repr__(self):
        return self.__name__


class Generic:
    """Abstract base class for generic types — PEP 484."""

    __slots__ = ()

    def __class_getitem__(cls, params):
        import typing
        return typing._generic_class_getitem(cls, params)

    def __init_subclass__(cls, *args, **kwargs):
        import typing
        return typing._generic_init_subclass(cls, *args, **kwargs)


# ── PEP 695 intrinsic helpers ────────────────────────────────────────────
# Called by the bytecode intrinsics (INTRINSIC_TYPEVAR, INTRINSIC_PARAMSPEC,
# INTRINSIC_SUBSCRIPT_GENERIC, ...) emitted for `class C[T]:`, `def f[T]()`,
# and `type X = ...`.  Keeping the construction here lets the interpreter side
# call a single positional helper per intrinsic.

def _intrinsic_typevar(name):
    return TypeVar._make(name)


def _intrinsic_paramspec(name):
    return ParamSpec._make(name)


def _intrinsic_typevartuple(name):
    return TypeVarTuple._make(name)


def _intrinsic_typevar_with_bound(name, evaluate_bound):
    return TypeVar._make(name, evaluate_bound=evaluate_bound)


def _intrinsic_typevar_with_constraints(name, evaluate_constraints):
    return TypeVar._make(name, evaluate_constraints=evaluate_constraints)


def _intrinsic_set_typeparam_default(typeparam, default):
    # CPython 3.14 `_Py_set_typeparam_default` stores the evaluator, not its
    # result.  `__default__` evaluates and caches it on first access.
    object.__setattr__(typeparam, "_default_value", _MISSING)
    object.__setattr__(typeparam, "_evaluate_default", default)
    return typeparam


def _intrinsic_subscript_generic(params):
    import typing
    if not isinstance(params, tuple):
        params = (params,)
    return typing._GenericAlias(typing.Generic, params)


def _intrinsic_typealias(args):
    # args is the (name, type_params, value) tuple the TYPEALIAS intrinsic
    # builds; `value` is the lazy evaluator (or the value itself).
    name, type_params, value = args
    if type_params is None:
        type_params = ()
    return TypeAliasType._from_evaluator(name, type_params, value)
