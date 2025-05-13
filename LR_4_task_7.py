import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1).ravel()

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)

    plt.plot(train_sizes, train_errors, 'o-', label='Тренувальний набір')
    plt.plot(train_sizes, val_errors, 's-', label='Валідаційний набір')
    plt.title(title)
    plt.xlabel("Кількість прикладів")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

# Лінійна модель
lin_reg = LinearRegression()
plot_learning_curve(lin_reg, X, y, "Крива навчання — лінійна модель")

# Поліноміальна модель
poly = PolynomialFeatures(degree=10, include_bias=False)
X_poly = poly.fit_transform(X)
lin_poly = LinearRegression()
plot_learning_curve(lin_poly, X_poly, y, "Крива навчання — поліном степеня 10")

# Поліноміальна модель 2-го ступеня
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
lin_poly2 = LinearRegression()
plot_learning_curve(lin_poly2, X_poly2, y, "Крива навчання — поліном степеня 2")
