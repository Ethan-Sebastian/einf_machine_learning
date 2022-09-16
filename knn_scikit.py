import pandas as pd
import numpy as np
import sklearn.neighbors as neigh
import sklearn.model_selection as modsel

column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris_data_set = pd.read_csv("iris.csv", names= column_names, header= None)

print(iris_data_set.head())                              # Die ersten fünf Samples
print(iris_data_set.tail())                              # Die letzten fünf Samples
print(iris_data_set.describe())

matrix_X = iris_data_set.iloc [0:149 ,  0:4]. values
vector_y = iris_data_set.iloc [0:149 ,  4]. values

X_train, X_test, y_train, y_test = modsel.train_test_split(matrix_X, vector_y, test_size=0.2, random_state=0,
                                                           stratify=vector_y)

classifier = neigh.KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(f"Genauigkeit: {accuracy}")

x = np.array([6.9, 2.9, 5.0, 1.0])
x_row_vector = x.reshape((1, 4))

predicted_class = classifier.predict(x_row_vector)
print(predicted_class)