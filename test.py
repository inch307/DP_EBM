a = {'1': 2, '3': 3}
b = [2, 3]

def A(l):
    a = 0
    for i in l.values():
        a+=i

    return a

print(A(a))
print(A(b))