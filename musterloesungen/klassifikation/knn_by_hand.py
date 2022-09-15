import numpy as np
import pandas as pd
import sklearn.model_selection as modsel
import statistics

column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
iris_data_set = pd.read_csv("iris.csv", names=column_names, header=None)
print(iris_data_set.head())
print(iris_data_set.tail())
print(iris_data_set.describe())

matrix_X = iris_data_set.iloc[0:150, 0:4].values
vector_y = iris_data_set.iloc[0:150, 4].values

# Daten in Trainings und Testdaten aufteilen (80:20).
X_train, X_test, y_train, y_test = modsel.train_test_split(matrix_X, vector_y, test_size=0.2, random_state=0,
                                                           stratify=vector_y)

number_of_neighbors = 5

error = 0
for index in range(len(y_test)):
    actual_class = y_test[index]
    sample = X_test[index, :]
    # Berechne Abstände zum Sample
    sample_row_vector = sample.reshape((1, 4))
    matrix_der_verbindungsvektoren = sample_row_vector - X_train
    laengen_verbindungsvektoren = np.linalg.norm(matrix_der_verbindungsvektoren, axis=1)
    # Bestimme die Klasse für den Sample
    tmp = np.copy(laengen_verbindungsvektoren)
    votes = []
    for neighbor in range(number_of_neighbors):
        index_minimale_laenge = np.argmin(tmp)
        klasse_nachbar = y_train[index_minimale_laenge]
        votes.append(klasse_nachbar)
        np.delete(tmp, index_minimale_laenge)
    # Bestimmte die Klasse die am häufigsten vorkommt.
    predicted_class = statistics.mode(votes)
    if actual_class != predicted_class:
        print(f"Vorhergesagte Klasse: {predicted_class}, Tatsächliche Klasse: {actual_class}")
        error = error + 1
print(error)
print(f"Genauigkeit (by hand): {1 - error / len(y_test)}")
