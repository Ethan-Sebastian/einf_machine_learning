import pandas as pd
import sklearn.svm as svm

column_names = ["x-coord.", "y-coord.", "class"]
svm_data_set = pd.read_csv("SVM.py", names= column_names, header= None)

print(svm_data_set.head())                              # Die ersten fünf Samples
print(svm_data_set.tail())                              # Die letzten fünf Samples
print(svm_data_set.describe())

matrix_X = svm_data_set.iloc [0:12, 0:3]
vector_y = svm_data_set.iloc [0:12, 3]