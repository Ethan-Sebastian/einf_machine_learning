import pandas as pd
import sklearn.svm as svm
import sklearn.model_selection as modsel

diabetes_data_set = pd.read_csv("diabetes.csv", header= "infer")
print(diabetes_data_set.head())
print(diabetes_data_set.tail())
print(diabetes_data_set.describe())

matrix_X = diabetes_data_set.iloc [0:768, 0:8]. values
vector_y = diabetes_data_set.iloc [0:768, 8]. values

svclassifier = svm.LinearSVC(max_iter=10000)
train_X, test_X, train_y, test_y = modsel.train_test_split(matrix_X, vector_y, test_size=0.20)

scores = modsel.cross_val_score(svclassifier, train_X, train_y, cv=3)
print(scores)

svclassifier.fit(train_X, train_y)
score = svclassifier.score(test_X, test_y)
print(f"Accuracy:{score}")