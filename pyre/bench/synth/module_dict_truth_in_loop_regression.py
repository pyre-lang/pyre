# pyre-check: selfcheck
# pyre-check: selfcheck-compiles=count_truthy
# Regression guard for a declined helper descent under the sub-walk driver.
#
# `if g:` on a module dict descends `truth_value`, which reaches
# `W_ModuleDictObject`'s `length` through the strategy's indirect call and
# from there `DictOperationGuard::new`, whose `push_roots` has no published
# address.  The descent records that residual call before the executor
# declines its symbolic funcbox.  Under the explicit sub-walk driver the
# callee's failure replays the parent CALL step, which used to take its
# rollback position only then -- past the callee's IR -- so the declined
# `push_roots` call stayed in the trace and the backend called the hash.


def count_truthy(g):
    n = 0
    for i in range(20000):
        if g:
            n += 1
    return n


assert count_truthy(globals()) == 20000
print("PASS")
