from __pypy__ import get_contextvar_context, set_contextvar_context
from _immutables_map import Map
from _pypy_generic_alias import GenericAlias
try:
    from __pypy__ import hidden_applevel
except ImportError:
    hidden_applevel = lambda f: f
# implementation taken from PEP-0567 https://www.python.org/dev/peps/pep-0567/

_NO_DEFAULT = object()
_TOKEN_NEW = object()


def _hexid(obj):
    return f"0x{id(obj):x}"


class Unsubclassable(type):
    def __new__(cls, name, bases, dct):
        for base in bases:
            if isinstance(base, Unsubclassable):
                # [3.15-spec] C types report tp_name (`_contextvars.ContextVar`).
                # lib_pypy interpolates only `base.__name__`.
                tp_name = base.__name__
                mod = getattr(base, "__module__", None)
                if mod:
                    tp_name = f"{mod}.{base.__name__}"
                raise TypeError(f"type '{tp_name}' is not an acceptable base type")
        return type.__new__(cls, name, bases, dict(dct))


class _Readonly:
    def __init__(self, attr, message=None):
        self.attr = attr
        self.message = message
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name
        if self.message is None:
            self.message = (
                f"attribute '{name}' of '{owner.__module__}.{owner.__name__}' "
                f"objects is not writable"
            )

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj, self.attr)

    def __set__(self, obj, value):
        raise AttributeError(self.message)

    def __delete__(self, obj):
        raise AttributeError(self.message)


def get_context():
    context = get_contextvar_context()
    if context is None:
        context = Context()
        set_contextvar_context(context)
    return context


class Context(metaclass=Unsubclassable):

    #_data: Map
    #_is_entered: bool

    def __init__(self, *args, **kwargs):
        # [3.15-spec] context_tp_new: "Context() does not accept any arguments",
        # still accepting **{}. lib_pypy: `def __init__(self)`.
        if args or kwargs:
            raise TypeError("Context() does not accept any arguments")
        self._data = Map()
        self._is_entered = False

    @hidden_applevel
    def run(self, callable, *args, **kwargs):
        if self._is_entered:
            raise RuntimeError(
                f'cannot enter context: {self} is already entered')

        # don't use get_context() here, to avoid creating a Context object
        _prev_context = get_contextvar_context()
        try:
            self._is_entered = True
            set_contextvar_context(self)
            return callable(*args, **kwargs)
        finally:
            set_contextvar_context(_prev_context)
            self._is_entered = False

    def copy(self):
        new = Context()
        new._data = self._data
        return new

    # Implement abstract Mapping.__getitem__
    def __getitem__(self, var):
        if not isinstance(var, ContextVar):
            raise TypeError("ContextVar key was expected")
        return self._data[var]

    # Implement abstract Mapping.__contains__
    def __contains__(self, var):
        if not isinstance(var, ContextVar):
            raise TypeError("ContextVar key was expected")
        return var in self._data

    # Implement abstract Mapping.__len__
    def __len__(self):
        return len(self._data)

    # Implement abstract Mapping.__iter__
    def __iter__(self):
        return iter(self._data)

    def get(self, key, default=None):
        if not isinstance(key, ContextVar):
            raise TypeError("ContextVar key was expected")
        try:
            return self._data[key]
        except KeyError:
            return default

    def keys(self):
        from collections.abc import KeysView
        return KeysView(self)

    def values(self):
        from collections.abc import ValuesView
        return ValuesView(self)

    def items(self):
        from collections.abc import ItemsView
        return ItemsView(self)

    def __eq__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        return dict(self.items()) == dict(other.items())


def copy_context():
    return get_context().copy()

class ContextVar(metaclass=Unsubclassable):

    def __new__(cls, *args, **kwargs):
        # [3.15-spec] contextvar_tp_new parses `"O|$O:ContextVar"` (name
        # positional-only, default keyword-only). lib_pypy: `__init__(self, name, *, default=_NO_DEFAULT)`.
        default = kwargs.pop("default", _NO_DEFAULT)
        if kwargs:
            if not args and set(kwargs) == {"name"}:
                raise TypeError(
                    "ContextVar() takes exactly 1 positional argument (0 given)"
                )
            unexpected = next(iter(kwargs))
            raise TypeError(
                f"ContextVar() got an unexpected keyword argument '{unexpected}'"
            )
        if len(args) == 0:
            raise TypeError(
                "ContextVar() takes exactly 1 positional argument (0 given)"
            )
        if len(args) > 1:
            raise TypeError(
                f"ContextVar() takes at most 1 positional argument ({len(args)} given)"
            )
        name = args[0]
        if not isinstance(name, str):
            raise TypeError("context variable name must be a str")
        hash(name)
        self = object.__new__(cls)
        self._name = name
        self._default = default
        return self

    def __init__(self, *args, **kwargs):
        pass

    name = _Readonly("_name", "readonly attribute")

    def get(self, *args, **kwargs):
        # [3.15-spec] clinic ContextVar.get is positional-only (`/`): extra
        # keywords are "takes no keyword arguments", extra positionals are
        # "get expected at most 1 argument, got N". lib_pypy:
        # `def get(self, default=_NO_DEFAULT)`.
        if kwargs:
            raise TypeError("ContextVar.get() takes no keyword arguments")
        if len(args) > 1:
            raise TypeError(f"get expected at most 1 argument, got {len(args)}")
        default = args[0] if args else _NO_DEFAULT
        # don't use get_context() here, to avoid creating a Context object
        context = get_contextvar_context()
        if context is not None:
            try:
                return context[self]
            except KeyError:
                pass

        if default is not _NO_DEFAULT:
            return default

        if self._default is not _NO_DEFAULT:
            return self._default

        # [3.15-spec] PyErr_SetObject(LookupError, var). lib_pypy: `raise LookupError`.
        raise LookupError(self)

    def set(self, value):
        context = get_context()

        data: Map = context._data
        try:
            old_value = data[self]
        except KeyError:
            old_value = Token.MISSING

        updated_data = data.set(self, value)
        context._data = updated_data
        return Token(context, self, old_value, _token_new=_TOKEN_NEW)

    def reset(self, token):
        # [3.15-spec] contextvar_reset checks Token before reading fields.
        # lib_pypy reads `token._used` first.
        if not isinstance(token, Token):
            raise TypeError(f"expected an instance of Token, got {token!r}")

        if token._used:
            # [3.15-spec] "%R has already been used once". lib_pypy:
            # "Token has already been used once".
            raise RuntimeError(f"{token!r} has already been used once")

        if token._var is not self:
            # [3.15-spec] "%R was created by a different ContextVar".
            # lib_pypy: "Token was created by a different ContextVar".
            raise ValueError(f"{token!r} was created by a different ContextVar")

        context = get_context()
        if token._context is not context:
            # [3.15-spec] "%R was created in a different Context".
            # lib_pypy: "Token was created in a different Context".
            raise ValueError(f"{token!r} was created in a different Context")

        if token._old_value is Token.MISSING:
            context._data = context._data.delete(token._var)
        else:
            context._data = context._data.set(token._var, token._old_value)

        token._used = True

    def __class_getitem__(self, item):
        return GenericAlias(self, item)

    def __repr__(self):
        default = ''
        if self._default is not _NO_DEFAULT:
            # [3.15-spec] contextvar_tp_repr writes repr(default).
            # lib_pypy interpolates `default={self._default}` (str).
            default = f"default={self._default!r} "
        return f"<ContextVar name={self.name!r} {default}at {_hexid(self)}>"


_allow_missing = True


class _TokenMissing(metaclass=Unsubclassable):
    def __new__(cls, *args, **kwargs):
        if not _allow_missing:
            raise TypeError("cannot create 'Token.MISSING' instances")
        return object.__new__(cls)

    def __repr__(self):
        return "<Token.MISSING>"


Token_MISSING = _TokenMissing()
_allow_missing = False
_TokenMissing.__name__ = "MISSING"
_TokenMissing.__qualname__ = "MISSING"
_TokenMissing.__module__ = "Token"


class Token(metaclass=Unsubclassable):
    MISSING = Token_MISSING

    def __new__(cls, *args, **kwargs):
        # [3.15-spec] token_tp_new: RuntimeError, and Token keeps object's
        # tp_init, so `Token.__init__(t)` is a no-op. lib_pypy constructs
        # Token from ContextVar.set via __init__.
        if kwargs.get("_token_new") is not _TOKEN_NEW:
            raise RuntimeError("Tokens can only be created by ContextVars")
        self = object.__new__(cls)
        self._context, self._var, self._old_value = args
        self._used = False
        return self

    var = _Readonly("_var")
    old_value = _Readonly("_old_value")

    def __repr__(self):
        return f"<Token {'used ' if self._used else ''}var={self._var} at {_hexid(self)}>"

    def __enter__(self):
        # [3.15-spec] Token is a context manager (3.14+). lib_pypy has neither.
        return self

    def __exit__(self, type, val, tb):
        self._var.reset(self)

    def __class_getitem__(self, item):
        return GenericAlias(self, item)

    __hash__ = None
