import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
# Load the diabetes dataset
diabetes = datasets.load_diabetes()
f, axarr = plt.subplots(5,2, sharex=True, sharey=True,figsize=(12,12))
for i in range(0,5):
    for j in range(0,2):
        # Use only one feature
        diabetes_X = diabetes.data[:, np.newaxis, i*2+j]
        print(diabetes_X.shape)
        # Split the data into training/testing sets
        diabetes_X_train = diabetes_X[:-20]
        diabetes_X_test = diabetes_X[-20:]
        
        # Split the targets into training/testing sets
        diabetes_y_train = diabetes.target[:-20]
        diabetes_y_test = diabetes.target[-20:]
        
        # Create linear regression object
        regr = linear_model.LinearRegression()
        
        # Train the model using the training sets
        regr.fit(diabetes_X_train, diabetes_y_train)
        
        # Plot outputs
        axarr[i,j].scatter(diabetes_X_test, diabetes_y_test,  color='red')
        axarr[i,j].scatter(diabetes_X_test, regr.predict(diabetes_X),  color='red')
        #axarr[i,j].plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue', linewidth=1)
        
plt.show()