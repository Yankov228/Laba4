import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as sm

data = np.loadtxt('data_regr_3.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

print("Mean absolute error:", round(sm.mean_absolute_error(y_test, y_pred), 2))
print("Mean squared error:", round(sm.mean_squared_error(y_test, y_pred), 2))
print("R2 score:", round(sm.r2_score(y_test, y_pred), 2))
