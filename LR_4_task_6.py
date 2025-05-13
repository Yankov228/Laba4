import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_poly_pred = poly_reg.predict(X_poly)

X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)

y_lin_plot = lin_reg.predict(X_plot)
y_poly_plot = poly_reg.predict(X_plot_poly)

plt.scatter(X, y, color='gray', label='Дані')
plt.plot(X_plot, y_lin_plot, label='Лінійна', color='blue')
plt.plot(X_plot, y_poly_plot, label='Поліноміальна', color='red')
plt.legend()
plt.title("Регресії")
plt.show()

print("Лінійна модель: y =", lin_reg.intercept_[0], "+", lin_reg.coef_[0][0], "* x")
print("Поліноміальна модель: y =", poly_reg.intercept_, "+", poly_reg.coef_)
