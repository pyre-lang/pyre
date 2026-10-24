"""PyPy-shaped BLAKE2 objects with object-owned native digest state.

The validation and object shape follow ``lib_pypy/_blake2/__init__.py``.
Only the RFC 7693 state machine lives in ``_hashlib._blake2_new`` so copying,
incremental updates, and GC ownership remain on the Python hash object rather
than in a TLS cache or global side table.
"""

from operator import index as _index

from _hashlib import _blake2_new as _new_state


class _Immutable(type):
    def __init__(cls, name, bases, namespace):
        type.__setattr__(cls, "_immutable_names", frozenset(namespace))
        type.__init__(cls, name, bases, namespace)

    def __setattr__(cls, name, value):
        qualname = ".".join((cls.__module__, cls.__name__))
        raise TypeError(
            "cannot set %r attribute of immutable type %r" % (name, qualname)
        )


_MISSING = object()


def _buffer_size(value):
    """Byte length of a buffer parameter, the `Py_buffer.len` the clinic
    converter measures.

    Not `len()`: a memoryview over a wider itemsize counts items, so
    `array('I', [0] * 5)` is 5 there and 20 bytes here, and only the byte
    count decides whether the salt fits.
    """
    if type(value) is bytes:
        return len(value)
    return memoryview(value).nbytes


def _make_blake_type(class_name, _salt_size, _person_size, _key_size,
                     _digest_size, max_offset, _block_size):
    class _Blake(metaclass=_Immutable):
        SALT_SIZE = _salt_size
        PERSON_SIZE = _person_size
        MAX_KEY_SIZE = _key_size
        MAX_DIGEST_SIZE = _digest_size

        def __new__(cls, *args, **kwargs):
            # Argument Clinic diagnoses a duplicate positional/keyword `data`
            # before unknown keywords, then diagnoses unknown keywords before
            # the data/string conflict. Python-function binding uses different
            # wording, so preserve CPython's public ordering explicitly.
            if len(args) > 1:
                raise TypeError(
                    "%s() takes at most 1 positional argument (%d given)" %
                    (class_name, len(args))
                )
            if args and "data" in kwargs:
                raise TypeError(
                    "argument for %s() given by name ('data') and position (1)" %
                    class_name
                )
            allowed = {
                "data", "digest_size", "key", "salt", "person", "fanout",
                "depth", "leaf_size", "node_offset", "node_depth",
                "inner_size", "last_node", "usedforsecurity", "string",
            }
            for keyword in kwargs:
                if keyword not in allowed:
                    raise TypeError(
                        "%s() got an unexpected keyword argument %r" %
                        (class_name, keyword)
                    )
            data = args[0] if args else kwargs.get("data", _MISSING)
            string = kwargs.get("string", _MISSING)
            if data is not _MISSING and string is not _MISSING:
                raise TypeError(
                    "'data' and 'string' are mutually exclusive and support "
                    "for 'string' keyword parameter is slated for removal in "
                    "a future version."
                )
            if data is _MISSING:
                if string is not _MISSING:
                    import warnings
                    warnings.warn(
                        "the 'string' keyword parameter is deprecated since "
                        "Python 3.15 and slated for removal in Python 3.19; "
                        "use the 'data' keyword parameter or pass the data "
                        "to hash as a positional argument instead",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    data = string
                else:
                    data = b""

            digest_size = kwargs.get("digest_size", _digest_size)
            key = kwargs.get("key", b"")
            salt = kwargs.get("salt", b"")
            person = kwargs.get("person", b"")
            fanout = kwargs.get("fanout", 1)
            depth = kwargs.get("depth", 1)
            leaf_size = kwargs.get("leaf_size", 0)
            node_offset = kwargs.get("node_offset", 0)
            node_depth = kwargs.get("node_depth", 0)
            inner_size = kwargs.get("inner_size", 0)
            last_node = kwargs.get("last_node", False)
            usedforsecurity = kwargs.get("usedforsecurity", True)

            # PyPy sets every integer parameter into the native parameter
            # block after range checking. operator.index matches the clinic
            # integer converters while accepting integer subclasses.
            digest_size = _index(digest_size)
            fanout = _index(fanout)
            depth = _index(depth)
            leaf_size = _index(leaf_size)
            node_offset = _index(node_offset)
            node_depth = _index(node_depth)
            inner_size = _index(inner_size)
            if not 1 <= digest_size <= cls.MAX_DIGEST_SIZE:
                raise ValueError(
                    "digest_size must be between 1 and %d bytes" %
                    cls.MAX_DIGEST_SIZE
                )
            # Salt and person are rejected before the tree parameters and the
            # key after them, the order lib_pypy/_blake2 sets each field in.
            # `blake2b(salt=b'x' * 17, fanout=256)` reports the salt.
            if _buffer_size(salt) > cls.SALT_SIZE:
                raise ValueError(
                    "maximum salt length is %d bytes" % cls.SALT_SIZE
                )
            if _buffer_size(person) > cls.PERSON_SIZE:
                raise ValueError(
                    "maximum person length is %d bytes" % cls.PERSON_SIZE
                )
            if not 0 <= fanout <= 255:
                raise ValueError("fanout must be between 0 and 255")
            if not 1 <= depth <= 255:
                raise ValueError("depth must be between 1 and 255")
            if leaf_size < 0:
                raise ValueError("value must be positive")
            if leaf_size > 0xFFFFFFFF:
                raise OverflowError("leaf_size is too large")
            if node_offset < 0:
                raise ValueError("value must be positive")
            if node_offset > max_offset:
                raise OverflowError("node_offset is too large")
            if not 0 <= node_depth <= 255:
                raise ValueError("node_depth must be between 0 and 255")
            if not 0 <= inner_size <= cls.MAX_DIGEST_SIZE:
                raise ValueError(
                    "inner_size must be between 0 and %d" %
                    cls.MAX_DIGEST_SIZE
                )
            if _buffer_size(key) > cls.MAX_KEY_SIZE:
                raise ValueError(
                    "maximum key length is %d bytes" % cls.MAX_KEY_SIZE
                )

            # Both clinic bool converters are observable through __bool__.
            bool(usedforsecurity)
            last_node = bool(last_node)
            self = object.__new__(cls)
            self._state = _new_state(
                class_name, data, digest_size, key, salt, person, fanout,
                depth, leaf_size, node_offset, node_depth, inner_size,
                last_node,
            )
            return self

        @property
        def name(self):
            return class_name

        @property
        def block_size(self):
            return _block_size

        @property
        def digest_size(self):
            return self._state.digest_size

        def update(self, data):
            self._state.update(data)

        def digest(self):
            return self._state.digest()

        def hexdigest(self):
            return self._state.hexdigest()

        def copy(self):
            other = object.__new__(type(self))
            other._state = self._state.copy()
            return other

        def __repr__(self):
            return "<%s.%s object at 0x%x>" % (
                type(self).__module__, type(self).__name__, id(self)
            )

    # The class statement above names both types `_Blake`; the factory's two
    # results have to answer with the name each was asked for.  `__module__`
    # needs no such repair -- the class body already read it off this file's
    # own `__name__`.
    type.__setattr__(_Blake, "__name__", class_name)
    type.__setattr__(_Blake, "__qualname__", class_name)
    return _Blake


blake2b = _make_blake_type("blake2b", 16, 16, 64, 64, (1 << 64) - 1, 128)
blake2s = _make_blake_type("blake2s", 8, 8, 32, 32, (1 << 48) - 1, 64)
