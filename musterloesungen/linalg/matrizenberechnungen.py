import numpy as np
import numpy.linalg as linalg

# Matrixeingabe
matrix_A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix_A)

# Transponieren
matrix_A_transponiert = matrix_A.T
# matrix_A_transponiert = matrix_A.transpose()
print(matrix_A_transponiert)

# Matrix mal Vektor
x = np.array([[1, 2, 3]]).T
# @ Operator: Matrix mal Vektor
b = matrix_A @ x
print(b)

# Matrix mal Matrix
matrix_B = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
# @ Operator: Matrix mal Matrix
matrix_C = matrix_A @ matrix_B
print("Matrix")
print(matrix_C)

# Inverse einer Matrix
matrix_T = np.array([[2, 1], [6, 4]])
matrix_T_invers = linalg.inv(matrix_T)
print(matrix_T_invers)
