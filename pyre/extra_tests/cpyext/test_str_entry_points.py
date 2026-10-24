# cpyext-fixture: cpyext_str
# cpyext-expect: cpyext-str-ok

# The `str` entry points, and the `%`-format engine three of them share.
#
# Every expectation was taken from CPython 3.14.6 running this same script
# against this same fixture, except where noted: `%p` answers with the
# platform's own spelling of a pointer, two entry points read an argument
# CPython never checks, which is not behaviour to match, and the `%d`
# operand TypeError uses the 3.15 `format argument:` prefix. Too-few
# operands use the 3.15 `(got N)` suffix.

import cpyext_str as m

class S(str):
    pass

def eq(name, got, want):
    assert got == want, '%s: got %r, want %r' % (name, got, want)

eq('concat', m.str_concat('ab', 'cd'), 'abcd')
eq('concat non-str', m.str_concat('ab', 3),
   ('TypeError', 'can only concatenate str (not "int") to str'))
# `PyUnicode_Format` is `%` between a format and its operands, which is what
# a Cython `"..." % x` compiles into.
eq('format tuple', m.str_format('%s=%d', ('n', 7)), 'n=7')
eq('format one', m.str_format('[%s]', 'x'), '[x]')
eq('format mapping', m.str_format('%(a)s', {'a': 1}), '1')
eq('format non-str format', m.str_format(3, ()),
   ('TypeError', 'must be str, not int'))
eq('format wrong operand', m.str_format('%d', 'x'),
   ('TypeError', 'format argument: %d requires a real number, not str'))
eq('format too few', m.str_format('%s %s', ('one',)),
   ('TypeError', 'not enough arguments for format string (got 1)'))

eq('append', m.str_append('ab', 'cd'), 'abcd')
eq('append_and_del', m.str_append_and_del('ab', 'cd'), 'abcd')

eq('substring', m.str_substring('abcdef', 1, 4), 'bcd')
eq('substring clamp', m.str_substring('abc', 1, 99), 'bc')
eq('substring empty', m.str_substring('abc', 2, 1), '')
eq('substring negative', m.str_substring('abc', -1, 2),
   ('IndexError', 'string index out of range'))

eq('join', m.str_join('-', ['a', 'b', 'c']), 'a-b-c')
eq('join non-str item', m.str_join('-', ['a', 2]),
   ('TypeError', 'sequence item 1: expected str instance, int found'))

eq('findchar forward', m.str_find_char('abcabc', ord('b'), 0, 6, 1), 1)
eq('findchar backward', m.str_find_char('abcabc', ord('b'), 0, 6, -1), 4)
eq('findchar absent', m.str_find_char('abc', ord('z'), 0, 3, 1), -1)
eq('findchar window', m.str_find_char('abcabc', ord('b'), 2, 6, 1), 4)

eq('contains', m.str_contains('abc', 'b'), True)
eq('contains absent', m.str_contains('abc', 'z'), False)
eq('contains non-str', m.str_contains('abc', 3),
   ('TypeError', "'in <string>' requires string as left operand, not int"))

eq('compare less', m.str_compare('a', 'b'), -1)
eq('compare equal', m.str_compare('a', 'a'), 0)
eq('compare greater', m.str_compare('b', 'a'), 1)
eq('compare non-str', m.str_compare('a', 3), ('TypeError', "Can't compare str and int"))
# The C string is read as ISO-8859-1, so it is a comparison of code points
# against bytes and never fails.
eq('compare ascii equal', m.str_compare_ascii('abc', 'abc'), 0)
eq('compare ascii less', m.str_compare_ascii('abc', 'abd'), -1)
eq('compare ascii shorter', m.str_compare_ascii('ab', 'abc'), -1)
eq('compare ascii longer', m.str_compare_ascii('abcd', 'abc'), 1)

eq('rich equal', m.str_rich_compare('a', 'a', 2), True)
eq('rich less', m.str_rich_compare('a', 'b', 0), True)
eq('equal', m.str_equal('ab', 'ab'), True)
eq('equal no', m.str_equal('ab', 'ac'), False)
eq('equal utf8', m.str_equal_utf8('été', 'été'.encode()), (True, True))
eq('equal utf8 no', m.str_equal_utf8('été', b'nope'), (False, False))

eq('ordinal', m.str_from_ordinal(0x1f600), '😀')
eq('ordinal ascii', m.str_from_ordinal(65), 'A')
eq('ordinal too big', m.str_from_ordinal(0x110000),
   ('ValueError', 'chr() arg not in range(0x110000)'))
eq('ordinal negative', m.str_from_ordinal(-1),
   ('ValueError', 'chr() arg not in range(0x110000)'))

# An exact str is answered with itself; a subclass instance is copied.
eq('from_object exact', m.str_from_object('abc'), ('abc', 'str', True))
eq('from_object subclass', m.str_from_object(S('abc')), ('abc', 'str', False))
eq('from_object other', m.str_from_object(3),
   ('TypeError', "Can't convert 'int' object to str implicitly"))

eq('intern', m.str_intern('a-name-nobody-else-uses'), ('a-name-nobody-else-uses', True))
eq('intern in place', m.str_intern_in_place('another-name-nobody-uses'),
   ('another-name-nobody-uses', True))

# The error handler is the interpreter's own, so every one it has is reachable.
eq('decode strict', m.str_decode_utf8(b'\xc3\xa9', None), 'é')
eq('decode strict invalid', m.str_decode_utf8(b'\xff', None),
   ('UnicodeDecodeError',
    "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte"))
eq('decode replace', m.str_decode_utf8(b'a\xffb', 'replace'), 'a�b')
eq('decode ignore', m.str_decode_utf8(b'a\xffb', 'ignore'), 'ab')

print('cpyext-str-ok')
