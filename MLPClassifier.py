import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

grzybki = pan.read_csv("agaricus-lepiota.data")
#grzybki = pan.read_csv("sample.data")

X = grzybki.drop(['0'], axis=1)
Y = grzybki.take([0], axis=1)
X = (X.apply(LabelEncoder().fit_transform))
Y = (Y.apply(LabelEncoder().fit_transform))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

network = MLPClassifier(hidden_layer_sizes=(3,3), max_iter = 50, solver='sgd')
network.fit(X_train, Y_train)
Y_predicted = network.predict(X_test)

Y_predicted = Y_predicted[:, np.newaxis]

error = np.mean((Y_predicted - Y_test)**2)
print(error)

f, axarr = plt.subplots(11, 2)
for i in range(2):
    for j in range(11):
        axarr[j, i].scatter(X_test.take([11*i+j], axis=1), Y_test,  color='red')
        axarr[j, i].plot(X_test.take([11*i+j], axis=1), Y_predicted, color='blue', linewidth=1)
plt.show()