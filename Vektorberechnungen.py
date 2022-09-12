import numpy as np

a = np.arange(4)
print(a)
c = a.reshape(2,2)
print(c)
b = np.array([8,4,25,0])
print(np.transpose(c))
print(np.transpose(b))
print(np.dot(a,b))
print(np.linalg.norm(b))
print(np.square(np.linalg.norm(b)))
print(np.dot(b,b))