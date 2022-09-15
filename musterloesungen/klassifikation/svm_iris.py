import pandas as pd
import sklearn.model_selection as modsel
import sklearn.svm as svm

column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
iris_data_set = pd.read_csv("iris.csv", names=column_names, header=None)
matrix_X = iris_data_set.iloc[0:150, 0:4].values
vektor_y = iris_data_set.iloc[0:150, 4].values

svclassifier = svm.LinearSVC(max_iter=10000)
train_X, test_X, train_y, test_y = modsel.train_test_split(matrix_X, vektor_y, test_size=0.20)

scores = modsel.cross_val_score(svclassifier, train_X, train_y, cv=3)
print(scores)

svclassifier.fit(train_X, train_y)
score = svclassifier.score(test_X, test_y)
print(f"Accuracy:{score}")
