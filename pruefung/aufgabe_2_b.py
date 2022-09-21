import pandas as pd
import sklearn.linear_model as linmodel

abalone_data_set = pd.read_csv("abalone.csv", header= None)

Matrix_X = abalone_data_set.iloc[0:4176, 1:8].values
vector_y = abalone_data_set.iloc[0:4176, 8].values


linear_regression = linmodel.LinearRegression()
linear_regression.fit(Matrix_X, vector_y)

print(linear_regression.coef_)