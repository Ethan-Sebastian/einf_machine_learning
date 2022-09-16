import numpy as np
import pandas as pd
import sklearn.linear_model as linmodel

diabetes_data_set = pd.read_csv("diabetes_regression.csv", header=None)
print(diabetes_data_set.head())
print(diabetes_data_set.tail())
print(diabetes_data_set.describe())

matrix_X_a = diabetes_data_set.iloc[0:442, 0:5].values
matrix_X_b = diabetes_data_set.iloc[0:442, 5:10].values
matrix_X = np.concatenate((matrix_X_a, matrix_X_b), axis=1)

vector_y = diabetes_data_set.iloc[0:442, 10].values

linear_regression = linmodel.LinearRegression()
linear_regression.fit(matrix_X, vector_y)

print(linear_regression.coef_)
print(linear_regression.intercept_)
