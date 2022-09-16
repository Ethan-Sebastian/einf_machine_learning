import pandas as pd
import sklearn.svm as svm

column_names = ["x-coord.", "y-coord.", "class"]
svm_data_set = pd.read_csv("abstrakt.csv", names= column_names, header= None)

print(svm_data_set.head())                              # Die ersten fünf Samples
print(svm_data_set.tail())                              # Die letzten fünf Samples
print(svm_data_set.describe())

matrix_X = svm_data_set.iloc [0:12, 0:2]. values
vector_y = svm_data_set.iloc [0:12, 2]. values

svclassifier = svm.SVC(kernel="linear")
svclassifier.fit(matrix_X, vector_y)
print("w = ", svclassifier.coef_)
print("b = ", svclassifier.intercept_)

x = [15, 10]
predicted_class = svclassifier.predict(x)
print(predicted_class)