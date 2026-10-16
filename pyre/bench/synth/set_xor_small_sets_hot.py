n = 0
for i in range(15000):
    a = {1, 2, 3, i % 11}
    b = {2, 3, 4, i % 13}
    n += len(a ^ b)
print(n)
