import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

error_rate = 0.02

def function(X_train, Y_train, X_test, n, i):
	
	network = MLPClassifier(hidden_layer_sizes=(n,n,n), max_iter = i)
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

my_error = 1
neurons = 2
iterations = 5
while my_error > error_rate:
	new_n = neurons
	new_i = iterations
	rand = random.randint(0, 5)
	if rand < 2:
		new_i = iterations + 5
	if rand == 2 and iterations > 3:
		new_i = iterations - 2
	if rand == 3 and new_n > 1:
		new_n = neurons - 1
	if rand > 3:
		new_n = neurons + 2

	new_error, Y_predicted = function(X_train, Y_train, X_test, new_n, new_i)

	print(new_error)
	if new_error < my_error:
		my_error = new_error
		neurons = new_n
		iterations = new_i

print("Neurony:")
print(neurons)
print("Maks. liczba iteracji")
print(iterations)

f, axarr = plt.subplots(11, 2)
for i in range(2):
    for j in range(11):
        temp = X_test.take([11*i+j], axis=1).to_numpy()
        axarr[j, i].scatter(temp, Y_test, color='red', marker='D', alpha=0.05)
        axarr[j, i].scatter(temp, Y_predicted, color='blue', marker='o', alpha=0.002)
plt.show()
