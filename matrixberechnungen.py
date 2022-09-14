import numpy as np

x = np.array([[1,2,4],[4,5,6],[7,8,9]])
print(x)
#y = np.array([[1,1,1],[1,1,1],[1,1,1]])
#print(np.matmul(x,y))
#print(np.transpose(x))
z = np.linalg.inv(x)
print(np.matmul(x,z))
