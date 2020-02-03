import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


def function(X_train, Y_train, X_test, i):
	
	network = MLPClassifier(hidden_layer_sizes=(i,i,i), max_iter = 50)
	network.fit(X_train, Y_train)
	Y_predicted = network.predict(X_test)

	Y_predicted = Y_predicted[:, np.newaxis]

	error = ((Y_predicted - Y_test)**2).mean()
	return error[0], Y_predicted
	


grzybki = pan.read_csv("agaricus-lepiota.data")
grzybki.replace('?', np.NaN)

X = grzybki.drop(['0'], axis=1)
Y = grzybki.take([0], axis=1)
X = (X.apply(LabelEncoder().fit_transform))
Y = (Y.apply(LabelEncoder().fit_transform))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#network = MLPClassifier(hidden_layer_sizes=(i,i,i), max_iter = 50)
#network.fit(X_train, Y_train)d
#Y_predicted = network.predict(X_test)

#Y_predicted = Y_predicted[:, np.newaxis]

#error = ((Y_predicted - Y_test)**2).mean()
#error_num = error[0]
#print(error[0])
#i = i + 1
my_error = 1
i = 1
while my_error > 0.02:
	my_error, Y_predicted = function(X_train, Y_train, X_test, i)
	print(my_error)
	i = i + 1

f, axarr = plt.subplots(11, 2)
for i in range(2):
    for j in range(11):
        temp = X_test.take([11*i+j], axis=1).to_numpy()
        axarr[j, i].scatter(temp, Y_test, color='red', marker='D', alpha=0.05)
        axarr[j, i].scatter(temp, Y_predicted, color='blue', marker='o', alpha=0.002)
plt.show()
