import pandas as pd
import sklearn.svm as svm

column_names = ["x1", "x2", "class"]
abstrakt_data_set = pd.read_csv("abstrakt.csv", names=column_names, header=None)
print(abstrakt_data_set.head())
print(abstrakt_data_set.tail())
print(abstrakt_data_set.describe())

matrix_X = abstrakt_data_set.iloc[0:12, 0:2].values
vektor_y = abstrakt_data_set.iloc[0:12, 2].values

svclassifier = svm.SVC(kernel="linear")
svclassifier.fit(matrix_X, vektor_y)
print("w = ", svclassifier.coef_)
print("b = ", svclassifier.intercept_)
print("Number of Support Vectors:", svclassifier.n_support_)
print("Support Vectors:", svclassifier.support_vectors_)
klasse = svclassifier.predict([[8, 8]])
print(klasse)
klasse = svclassifier.predict([[14, 14]])
print(klasse)
