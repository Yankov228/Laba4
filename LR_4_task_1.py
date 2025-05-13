import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3.0, 1.0, 1.8, 1.9])

coeffs = np.polyfit(x, y, deg=4)

p = np.poly1d(coeffs)

x_vals = np.linspace(0.05, 0.75, 500)
y_vals = p(x_vals)

plt.plot(x_vals, y_vals, label="Інтерполяційний поліном", color='blue')
plt.scatter(x, y, color='red', label="Точки (x, y)")
plt.title("Інтерполяція поліномом 4-го степеня")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

y_02 = p(0.2)
y_05 = p(0.5)

print(f"f(0.2) ≈ {y_02:.4f}")
print(f"f(0.5) ≈ {y_05:.4f}")
