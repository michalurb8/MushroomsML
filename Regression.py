import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder

grzybki = pan.read_csv("agaricus-lepiota.data")

X = grzybki.drop(['0'], axis=1)
Y = grzybki.take([0], axis=1)
X = (X.apply(LabelEncoder().fit_transform))
Y = (Y.apply(LabelEncoder().fit_transform))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
Y_predicted = regr.predict(X_test)

error = np.mean((Y_predicted - Y_test)**2)
print(error)

f, axarr = plt.subplots(11, 2)
for i in range(2):
    for j in range(11):
        axarr[j, i].scatter(X_test.take([11*i+j], axis=1), Y_test,  color='red')
        axarr[j, i].plot(X_test.take([11*i+j], axis=1), Y_predicted, color='blue', linewidth=1)
plt.show()