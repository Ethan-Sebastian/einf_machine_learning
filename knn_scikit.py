import pandas as pd
import sklearn.neigbours as neigh

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data_set = pd.read_csv("iris.csv", names= column_names, header= None)

print(iris_data_set.head())
print(iris_data_set.tail())
print(iris_data_set.describe())

matrix_X = iris_data_set.iloc [0:150 ,  0:4]. values
vector_y = iris_data_set.iloc [0:150 ,  4]. values
