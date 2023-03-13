import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv("/Users/artempilecki/Desktop/kc_house_data.csv", delimiter=",")

# Выбор входных параметров и выходного параметра
X = data[["bedrooms", "bathrooms", "sqft_living"]].values
y = data["price"].values

# Нормализация данных
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    norm_data = (data - min_val) / (max_val - min_val)
    return norm_data

X_norm = normalize(X)

# Расчет минимального, максимального и среднего значений для каждого параметра
for i, col in enumerate(["bedrooms", "bathrooms", "sqft_living"]):
    print(f"Column {i}: {col}")
    print(f"Min: {np.min(X_norm[:, i])}")
    print(f"Max: {np.max(X_norm[:, i])}")
    print(f"Mean: {np.mean(X_norm[:, i])}\n")

# Функция ошибки
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Функция расчета градиента
def gradient(X, y, w):
    N = len(y)
    y_pred = np.dot(X, w)
    grad = np.dot(X.T, (y_pred - y)) / N
    return grad

# Функция градиентного спуска
def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    N, m = X.shape
    w = np.zeros(m)
    for i in range(num_iterations):
        grad = gradient(X, y, w)
        w = w - learning_rate * grad
    return w

# Обучение модели
w = gradient_descent(X_norm, y)

# Вывод ошибки обученной модели
y_pred = np.dot(X_norm, w)
print(f"MSE: {mse(y, y_pred)}\n")

# Графики зависимости
for i, col in enumerate(["bedrooms", "bathrooms", "sqft_living"]):
    plt.scatter(X_norm[:, i], y, label="Data")
    plt.plot(X_norm[:, i], np.dot(X_norm, w), color="red", label="Regression")
    plt.xlabel(col)
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# Функциональный вид обученной модели
print(f"Price = {w[0]:.2f} + {w[1]:.2f}*bedrooms + {w[2]:.2f}*bathrooms + {w[3]:.2f}*sqft_living\n")

# Расчет стоимости квартиры
bedrooms = 3
bathrooms = 2
sqft_living = 2000
price = w[0] + w[1]*normalize(bedrooms) + w[2]*normalize(bathrooms) + w[3]*normalize(sqft_living)
print(f"Estimated price: {price:.2f}")