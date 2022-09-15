import pandas as pd
import sklearn.neural_network as nn
import sklearn.model_selection as modsel
# install Pillow
import PIL.Image as im

# install opencv-python
import cv2

mnist_training_set = pd.read_csv("mnist_train_100.csv", header=None)

X_train = mnist_training_set.iloc[:, 1:785].values
y_train = mnist_training_set.iloc[:, 0].values

multilayer_perceptron = nn.MLPClassifier()
multilayer_perceptron.fit(X_train, y_train)

scores = modsel.cross_val_score(multilayer_perceptron, X_train, y_train, cv=5)
print(scores)

mnist_test_set = pd.read_csv("mnist_test_100.csv", header=None)
X_test = mnist_training_set.iloc[:, 1:785].values
y_test = mnist_training_set.iloc[:, 0].values
score = multilayer_perceptron.score(X_test, y_test)

size = (28, 28)
image1 = "eins.png"
bild = im.open(image1)
bild_resized = bild.resize(size, im.LANCZOS)
bild_resized.save("eins_resized.png", "PNG")

bild = cv2.imread("eins_resized.png", 0)
bild = bild.ravel().reshape((1, 784))
prediction = multilayer_perceptron.predict(bild)
print(prediction)
