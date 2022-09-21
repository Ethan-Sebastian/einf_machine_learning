import numpy as np
import pandas as pd
import sklearn.neighbors as neigh
import sklearn.model_selection as modsel
import sklearn.svm as svm

abalone_data_set = pd.read_csv("abalone.csv", names= None, header= None)

Matrix_X = abalone_data_set.iloc[0:4176, 1:8].values
vector_y = abalone_data_set.iloc[0:4176, 8].values

X_train, X_test, y_train, y_test = modsel.train_test_split(Matrix_X, vector_y, test_size=0.2, random_state=0)

# k-Nearest-Neigbor

knn_classifier = neigh.KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)

knn_accuracy = knn_classifier.score(X_test, y_test)
print(f"Genauigkeit KNN mit k=3: {knn_accuracy}")

# Nearest Centroid

centroid_classifier = neigh.NearestCentroid()
centroid_classifier.fit(X_train, y_train)

centroid_accuracy = centroid_classifier.score(X_test, y_test)
print(f"Genauigkeit mit Nearest Centroid: {centroid_accuracy}")

# SVM

svm_classifier = svm.SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)

svm_accuracy = svm_classifier.score(X_test, y_test)
print(f"Genauigkeit mit SVM: {svm_accuracy}")