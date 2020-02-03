import pandas as pan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

grzybki = pan.read_csv("agaricus-lepiota.data")
grzybki.replace('?', np.NaN)


X = grzybki.drop(['0'], axis=1)
Y = grzybki.take([0], axis=1)
X = (X.apply(LabelEncoder().fit_transform))
Y = (Y.apply(LabelEncoder().fit_transform))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.count)

regr = LinearRegression()
regr.fit(X_test, Y_test)
Y_predicted = regr.predict(X_test)

error = np.mean((Y_predicted - Y_test)**2)
print(error)


f, axarr = plt.subplots(11, 2)
for i in range(2):
    for j in range(11):
        temp = X_test.take([11*i+j], axis=1).to_numpy()
        axarr[j, i].scatter(temp, Y_test, color='red', marker='D', alpha=0.05)
        axarr[j, i].scatter(temp, Y_predicted, color='blue', marker='.', alpha=0.01)
plt.show()