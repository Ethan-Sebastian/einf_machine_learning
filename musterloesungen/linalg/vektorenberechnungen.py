import numpy as np
# linalg = lineare algebra
import numpy.linalg as linalg

# Listen vs Array: Python-Listen erlauben unterschiedliche Datentypen.
# Arrays sind homogen und erlauben nur einen Datentyp.

# Vektor erstellen
vektor_a = np.array([1, 2, 3, 4])

# Dimension
dimension = np.shape(vektor_a)
print(f"Dimension: {dimension}")

# Transponieren
vektor_b = vektor_a.transpose()
dimension = np.shape(vektor_b)
print(f"Dimension: {dimension}")

# Skalarprodukt
vektor_c = np.array([5, 6, 7, 8])
skalarprodukt = np.dot(vektor_c, vektor_a)
print(f"Skalarprodukt: {skalarprodukt}")

# norm: Betrag eines Vektors
norm = linalg.norm(vektor_a)
print(f"Norm von {vektor_a}: {norm}")
