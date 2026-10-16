# Loop body raises UnboundLocalError from a deleted local and counts
# handler visits; the count must equal N. An abort that resumes past
# the raising LOAD_FAST_CHECK silently drops one increment per abort.
N = 100000
def run():
    c = 0
    e = 1
    del e
    for i in range(N):
        try:
            e
        except NameError:
            c += 1
    print("c =", c)
run()
