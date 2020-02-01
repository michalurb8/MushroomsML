import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

grzybki = pan.read_csv("agaricus-lepiota.data")
#grzybki = pan.read_csv("probka")

X = grzybki.drop(['0'], axis=1)
Y = grzybki.take([0], axis=1)
X = (X.apply(LabelEncoder().fit_transform))
Y = (Y.apply(LabelEncoder().fit_transform))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#print(X_train); print(); print(X_test); print(); print(Y_train); print(); print(Y_test); print()

network = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter = 50, solver='sgd')
network.fit(X_train, Y_train)
Y_predicted = network.predict(X_test)

Y_predicted=Y_predicted[:, np.newaxis]

error = np.mean((Y_predicted - Y_test)**2)
print(error)

f, axarr = plt.subplots()
axarr.scatter(X_test.take([0], axis=1), Y_test,  color='red')
axarr.plot(X_test, Y_predicted, color='blue', linewidth=1)
        
#plt.show()