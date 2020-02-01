import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

grzybki = pan.read_csv("agaricus-lepiota.data")

X = grzybki.drop(['0'], axis=1)
Y = grzybki.take([0], axis=1)
X = (X.apply(LabelEncoder().fit_transform))
Y = (Y.apply(LabelEncoder().fit_transform))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

#print(X_train); print(); print(X_test); print(); print(Y_train); print(); print(Y_test); print()

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
Y_predicted = regr.predict(X_test)
print(Y_test.shape)
print(Y_predicted.shape)

error = np.mean((Y_predicted - Y_test)**2)
print(error)

#f, axarr = plt.subplots()
#axarr.scatter(X_test.take([0], axis=1), Y_test,  color='red')
#axarr.plot(X_test, regr.predict(X_test), color='blue', linewidth=1)
#plt.show()