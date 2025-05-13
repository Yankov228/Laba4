import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import sklearn.metrics as sm

data = np.loadtxt('data_multivar_regr.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

print("Linear Regressor performance:")
print("MAE:", sm.mean_absolute_error(y_test, y_pred_linear))
print("MSE:", sm.mean_squared_error(y_test, y_pred_linear))
print("R2:", sm.r2_score(y_test, y_pred_linear))

polynomial = PolynomialFeatures(degree=10)
X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.transform(X_test)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

datapoint = [[7.75, 6.35, 5.56]]
datapoint_poly = polynomial.transform(datapoint)

print("\nPrediction for datapoint:")
print("Linear:", linear_regressor.predict(datapoint))
print("Polynomial:", poly_model.predict(datapoint_poly))
