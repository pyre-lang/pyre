# CALL_KW binding TypeErrors use `func.qualname` (`PyCode.funcrun` /
# `funcrun_obj` pass it to `parse_into_scope`). A method and a nested
# function therefore print as `C.m()` / `outer.<locals>.inner()`, not the
# bare `co_name`.
N = 300


class C:
    def m(self, a):
        return a


def outer():
    def inner(a):
        return a

    return inner


def main():
    obj = C()
    nested = outer()
    method_msg = None
    nested_msg = None
    for i in range(N):
        try:
            obj.m(zz=i)
        except TypeError as e:
            method_msg = str(e)
        try:
            nested(zz=i)
        except TypeError as e:
            nested_msg = str(e)
    print(method_msg)
    print(nested_msg)


main()
