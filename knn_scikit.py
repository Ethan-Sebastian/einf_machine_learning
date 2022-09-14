import pandas as pd
import sklearn.neighbors as neigh

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data_set = pd.read_csv("iris.csv", names= column_names, header= None)

print(iris_data_set.head())                              # Die ersten fünf Samples
print(iris_data_set.tail())                              # Die letzten fünf Samples
print(iris_data_set.describe())

matrix_X = iris_data_set.iloc [0:149 ,  0:4]. values
vector_y = iris_data_set.iloc [0:149 ,  4]. values

classifier = neigh.KNeighborsClassifier(3)
classifier.fit(matrix_X, vector_y)

x = [5.9, 3.0, 5.1, 1.8]
predicted_class = classifier.predict(x)
print(predicted_class)