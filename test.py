import numpy as np
import time

len = 100000

a = np.random.rand(len)
b = np.zeros_like(a)
l = np.zeros_like(a)

lst = [i for i in range(len)]

t1 = time.time()
b[lst] = l[lst] -1 + (1/(1+np.exp(a[lst])))
t2 = time.time()

print(t2-t1)

t1 = time.time()
b = l -1 + (1/(1+np.exp(a)))
t2 = time.time()

print(t2-t1)