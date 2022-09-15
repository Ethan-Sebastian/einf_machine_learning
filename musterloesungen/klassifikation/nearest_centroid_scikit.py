import numpy as np
import pandas as pd
import sklearn.neighbors as neigh
import sklearn.model_selection as modsel

# Spaltennamen definieren
column_names = ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
# Daten aus der CSV-Datei laden
iris_data_set = pd.read_csv("iris.csv", names=column_names, header=None)
# Die ersten und letzten Zeilen der Daten ausgeben, Statistik der Daten ausgeben
print(iris_data_set.head())
print(iris_data_set.tail())
print(iris_data_set.describe())

# Samples aus den Daten "extrahieren": 150 Zeilen und 4 Features
matrix_X = iris_data_set.iloc[0:150, 0:4].values
# Klassen aus den Daten "extrahieren": 150 Zeilen und 1 Klasse
vector_y = iris_data_set.iloc[0:150, 4].values

# Daten in Trainings und Testdaten aufteilen (80:20).
X_train, X_test, y_train, y_test = modsel.train_test_split(matrix_X, vector_y, test_size=0.2, random_state=0,
                                                           stratify=vector_y)

# Nearest Centroid Classifier
classifier = neigh.NearestCentroid()
# Daten zum Lernen übergeben.
classifier.fit(X_train, y_train)

# Teste die Samples mit dem Klassifizierer
# Vergleiche Vorhersage mit dem tatsächlichen Resultat
accuracy = classifier.score(X_test, y_test)
print(f"Genauigkeit: {accuracy}")

# Nimm Test-Vektor für die Vorhersage
x = np.array([5.84, 3.05, 3.76, 1.20])
x_row_vector = x.reshape((1, 4))
predicted_class = classifier.predict(x_row_vector)
print(f"Klassifikation: {predicted_class}")
