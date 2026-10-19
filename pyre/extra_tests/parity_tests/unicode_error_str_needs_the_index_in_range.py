# pyre-check: pypy-diverges: pins 3.14's range-form fallback when the offending
# index lies outside the object; pypy3 reads the object unguarded, so the same
# calls raise IndexError, and a negative start reads from the end instead.
#
# CPython-suite gap: `test_codeccallbacks` and `test_exceptions` build
# `UnicodeDecodeError` / `UnicodeEncodeError` / `UnicodeTranslateError` by hand
# and check the attributes and the ordinary one-character message, but every
# instance they stringify has its `start` inside the object.  Nothing there
# stringifies one whose `start` is past the end, or negative, so nothing notices
# a `__str__` that reports a character it could not read.
#
# parity-tests reason: all three `__str__` implementations take the
# single-character message only when the slice is genuinely inside the object --
# `start >= 0 and start < len and end >= 0 and end <= len and end == start + 1`
# -- and otherwise report the range form.  An implementation that keeps the
# single-character shape and substitutes a placeholder for the character it
# could not read prints a message no other implementation produces, and it does
# so only for hand-built instances, which no codec round trip reaches.
#
# unicode_error_str_null_slots_differ_from_none:
# `test_exceptions` stringifies a constructed UnicodeDecodeError and never deletes
# encoding, object, or reason. UnicodeDecodeError_str returns empty when object is NULL.
# PyObject_Str of a NULL encoding or reason spells `<NULL>`. A stored None is present:
# object None is TypeError from check_unicode_error_attribute, encoding/reason None
# stringify as `None`. PyMember_SetOne clears `_Py_T_OBJECT` to NULL.
def a_decode_error_past_the_end_reports_a_range():
    exc = UnicodeDecodeError('utf-8', b'', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't decode bytes in position 0-0: bad", str(exc)
    exc = UnicodeDecodeError('utf-8', b'ab', 5, 6, 'bad')
    assert str(exc) == "'utf-8' codec can't decode bytes in position 5-5: bad", str(exc)


def a_negative_decode_start_reports_a_range():
    exc = UnicodeDecodeError('utf-8', b'ab', -1, 0, 'bad')
    assert str(exc) == "'utf-8' codec can't decode bytes in position -1--1: bad", str(exc)


def an_in_range_decode_byte_still_names_the_byte():
    exc = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't decode byte 0xff in position 0: bad", str(exc)
    exc = UnicodeDecodeError('utf-8', b'ab', 1, 2, 'bad')
    assert str(exc) == "'utf-8' codec can't decode byte 0x62 in position 1: bad", str(exc)


def an_encode_error_past_the_end_reports_a_range():
    exc = UnicodeEncodeError('utf-8', '', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't encode characters in position 0-0: bad", str(exc)
    exc = UnicodeEncodeError('utf-8', 'ab', 5, 6, 'bad')
    assert str(exc) == "'utf-8' codec can't encode characters in position 5-5: bad", str(exc)
    exc = UnicodeEncodeError('utf-8', 'ab', -1, 0, 'bad')
    assert str(exc) == "'utf-8' codec can't encode characters in position -1--1: bad", str(exc)


def an_in_range_encode_character_still_names_the_character():
    exc = UnicodeEncodeError('utf-8', '\ud800', 0, 1, 'bad')
    assert str(exc) == "'utf-8' codec can't encode character '\\ud800' in position 0: bad", str(exc)


def a_translate_error_past_the_end_reports_a_range():
    exc = UnicodeTranslateError('', 0, 1, 'bad')
    assert str(exc) == "can't translate characters in position 0-0: bad", str(exc)
    exc = UnicodeTranslateError('ab', 5, 6, 'bad')
    assert str(exc) == "can't translate characters in position 5-5: bad", str(exc)


def an_in_range_translate_character_still_names_the_character():
    exc = UnicodeTranslateError('a', 0, 1, 'bad')
    assert str(exc) == "can't translate character '\\x61' in position 0: bad", str(exc)


a_decode_error_past_the_end_reports_a_range()
a_negative_decode_start_reports_a_range()
an_in_range_decode_byte_still_names_the_byte()
an_encode_error_past_the_end_reports_a_range()
an_in_range_encode_character_still_names_the_character()
a_translate_error_past_the_end_reports_a_range()
an_in_range_translate_character_still_names_the_character()

def unicode_error_str_null_slots_differ_from_none():
    def decode_deleted_object_is_empty():
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        del exc.object
        assert exc.object is None
        assert str(exc) == "", str(exc)


    def decode_assigned_none_object_typeerrors():
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        exc.object = None
        try:
            str(exc)
        except TypeError as err:
            assert str(err) == "UnicodeError 'object' attribute must be a bytes", str(err)
        else:
            raise AssertionError("expected TypeError")


    def decode_deleted_encoding_spells_null():
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        del exc.encoding
        assert (
            str(exc) == "'<NULL>' codec can't decode byte 0xff in position 0: bad"
        ), str(exc)


    def decode_assigned_none_encoding_spells_none():
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        exc.encoding = None
        assert (
            str(exc) == "'None' codec can't decode byte 0xff in position 0: bad"
        ), str(exc)


    def decode_deleted_reason_spells_null():
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        del exc.reason
        assert (
            str(exc) == "'utf-8' codec can't decode byte 0xff in position 0: <NULL>"
        ), str(exc)


    def decode_assigned_none_reason_spells_none():
        exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
        exc.reason = None
        assert (
            str(exc) == "'utf-8' codec can't decode byte 0xff in position 0: None"
        ), str(exc)


    def encode_deleted_object_is_empty():
        exc = UnicodeEncodeError("utf-8", "é", 0, 1, "bad")
        del exc.object
        assert str(exc) == "", str(exc)


    def encode_assigned_none_object_typeerrors():
        exc = UnicodeEncodeError("utf-8", "é", 0, 1, "bad")
        exc.object = None
        try:
            str(exc)
        except TypeError as err:
            assert str(err) == "UnicodeError 'object' attribute must be a string", str(err)
        else:
            raise AssertionError("expected TypeError")


    def translate_deleted_reason_spells_null():
        exc = UnicodeTranslateError("é", 0, 1, "bad")
        del exc.reason
        assert (
            str(exc) == "can't translate character '\\xe9' in position 0: <NULL>"
        ), str(exc)


    def translate_assigned_none_reason_spells_none():
        exc = UnicodeTranslateError("é", 0, 1, "bad")
        exc.reason = None
        assert str(exc) == "can't translate character '\\xe9' in position 0: None", str(exc)


    decode_deleted_object_is_empty()
    decode_assigned_none_object_typeerrors()
    decode_deleted_encoding_spells_null()
    decode_assigned_none_encoding_spells_none()
    decode_deleted_reason_spells_null()
    decode_assigned_none_reason_spells_none()
    encode_deleted_object_is_empty()
    encode_assigned_none_object_typeerrors()
    translate_deleted_reason_spells_null()
    translate_assigned_none_reason_spells_none()
unicode_error_str_null_slots_differ_from_none()
print('OK')
