# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import csv
import io
from statistics import mean
from sklearn import preprocessing
import random


# Чтение данных из файла
def Column_select(columns_data, n):
    columns_data = columns_data.iloc[:, [n]]
    return columns_data


data = pd.read_csv("kc_house_data.csv")
bedrooms = Column_select(data, 3)
bathrooms = Column_select(data, 4)
sqft_living = Column_select(data, 5)
price = Column_select(data, 2)


# Нормализация данных MinMaX
def Min_Max_norm(data_for_norm, column_name):
    norm_data = Min_Max.fit_transform(data_for_norm)
    norm_Min_MAx = pd.DataFrame(norm_data, columns=column_name)
    return norm_Min_MAx


# Создание объекта
Min_Max = preprocessing.MinMaxScaler(feature_range=(-1, 1))

# Название столбцов
Column_bedrooms = bedrooms.columns
Column_bathrooms = bathrooms.columns
Column_sqft_living = sqft_living.columns
Column_price = price.columns

# Нормализация данных
bedrooms_norm = Min_Max_norm(bedrooms, Column_bedrooms)
bathrooms_norm = Min_Max_norm(bathrooms, Column_bathrooms)
sqft_living_norm = Min_Max_norm(sqft_living, Column_sqft_living)
price_norm = Min_Max_norm(price, Column_price)


# Создание массивов
def data_to_array(array):
    y = len(array)

    list = []
    for i in range(y):
        for j in array[i]:
            list.append(j)
    return list


x1_norm = data_to_array(np.array(bedrooms_norm))
x2_norm = data_to_array(np.array(bathrooms_norm))
x3_norm = data_to_array(np.array(sqft_living_norm))
y_norm = data_to_array(np.array(price_norm))


# Вывод значений после нормализации
def test_norm(array, name):
    Value = sorted(array)
    Max = Value[-1]
    Min = Value[0]
    Mean = mean(Value)
    print("Данные: " + name)
    print("Max: ", Max)
    print("Min: ", Min)
    print("Mean: ", Mean)


test_norm(x1_norm, "x1 (bedrooms)")
test_norm(x2_norm, "x2 (bathrooms)")
test_norm(x3_norm, "x3 (sqft_living)")
test_norm(y_norm, "y (price)")

# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y_train, y_test = train_test_split(x1_norm, x2_norm, x3_norm,
                                                                                            y_norm, test_size=0.30,
                                                                                            random_state=0)

# Генерация случайных значений для коэффициентов


# Опредение первоначальных коэффициентов
# линейной регрессии (случайная генерация)
a0 = random.uniform(-1, 1)
a1 = random.uniform(-1, 1)
a2 = random.uniform(-1, 1)
a3 = random.uniform(-1, 1)

# Сгенерированные коэффициенты
print("a0 = " + str(round(a0, 3)))
print("a1 = " + str(round(a1, 3)))
print("a2 = " + str(round(a2, 3)))
print("a3 = " + str(round(a3, 3)))


# Линейная функция для трех входных параметров
def lin_func(x1, x2, x3):
    y = a0 + a1 * x1 + a2 * x2 + a3 * x3
    return y


# Функция тестирования и визуализации линейной функции
def test_lin_func(x1_res, x2_res, x3_res, y_res):
    # Вспомогательные массивы для построение линейной функции
    x1seq = np.linspace(np.min(x1_res), np.max(x1_res), 100)
    x2seq = np.linspace(np.min(x2_res), np.max(x2_res), 100)
    x3seq = np.linspace(np.min(x3_res), np.max(x3_res), 100)

    # График данных
    for i in range(len(y_res)):
        plt.plot(x1_res[i], y_res[i], 'ro')
    # График линейной функции
    plt.plot(x1seq, lin_func(x1seq, x2seq, x3seq), 'b')
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.show()

    # График данных
    for i in range(len(y_res)):
        plt.plot(x2_res[i], y_res[i], 'ro')
    # График линейной функции
    plt.plot(x2seq, lin_func(x1seq, x2seq, x3seq), 'b')
    plt.xlabel("x2")
    plt.ylabel("y")
    plt.show()

    # График данных
    for i in range(len(y_res)):
        plt.plot(x3_res[i], y_res[i], 'ro')
    # График линейной функции
    plt.plot(x3seq, lin_func(x1seq, x2seq, x3seq), 'b')
    plt.xlabel("x3")
    plt.ylabel("y")
    plt.show()


# Тест исходной НЕ ОБУЧЕННОЙ модели
test_lin_func(x1_test, x2_test, x3_test, y_test)


# Функция расчёта СРЕДНЕЙ КВАДРАТИЧНОЙ ошибки (MSE)
def err_func(x1_res, x2_res, x3_res, y_res):
    # Сумма ошибок
    O_func = 0
    for i in range(len(y_res)):
        # Квадратичная разница по модулю между реальным значением и рассчитанным
        O_func += math.pow((y_res[i] - lin_func(x1_res[i], x2_res[i], x3_res[i])), 2)
    # Деление суммы ошибок на количество рассчитанных ситуаций и рассчет корня получившихся значений
    MSE = O_func / len(y_res)

    return MSE


# Тест исходной НЕ ОБУЧЕННОЙ модели
print("MSE = " + str(err_func(x1_test, x2_test, x3_test, y_test)))

# Количество итераций обучения
I_count = 1000
# Константа скорости обучения
learn_step = 0.001

#Обучение модели при помощи градиентного спуска
for i in range(I_count):

    n = len(y_train)

    # Сумма для частных производных
    O_sum = 0
    O_sum_x1 = 0
    O_sum_x2 = 0
    O_sum_x3 = 0

    for i in range(n):
        O_sum += y_train[i] - (a0 + a1 * x1_train[i] + a2 * x2_train[i] + a3 * x3_train[i])
        O_sum_x1 += (y_train[i] - (a0 + a1 * x1_train[i] + a2 * x2_train[i] + a3 * x3_train[i])) * x1_train[i]
        O_sum_x2 += (y_train[i] - (a0 + a1 * x1_train[i] + a2 * x2_train[i] + a3 * x3_train[i])) * x2_train[i]
        O_sum_x3 += (y_train[i] - (a0 + a1 * x1_train[i] + a2 * x2_train[i] + a3 * x3_train[i])) * x3_train[i]

    # Частная производная функции СРЕДНЕЙ КВАДРАТИЧНОЙ ошибки (MSE) по a0
    dOda0 = (-2 / n) * O_sum
    a0_new = a0 - learn_step * dOda0
    # Частная производная функции СРЕДНЕЙ КВАДРАТИЧНОЙ ошибки (MSE) по a1
    dOda1 = (-2 / n) * O_sum_x1
    a1_new = a1 - learn_step * dOda1
    # Частная производная функции СРЕДНЕЙ КВАДРАТИЧНОЙ ошибки (MSE) по a2
    dOda2 = (-2 / n) * O_sum_x2
    a2_new = a2 - learn_step * dOda2
    # Частная производная функции СРЕДНЕЙ КВАДРАТИЧНОЙ ошибки (MSE) по a3
    dOda3 = (-2 / n) * O_sum_x3
    a3_new = a3 - learn_step * dOda3

    a0 = a0_new
    a1 = a1_new
    a2 = a2_new
    a3 = a3_new

# Тест ОБУЧЕННОЙ модели
test_lin_func(x1_test, x2_test, x3_test, y_test)
print("MSE = " + str(err_func(x1_test, x2_test, x3_test, y_test)))

# Итоговый функциональный вид обученной модели
print("y = " + str(round(a0, 3)) + " + " + str(round(a1, 3)) + " * x1 + " + str(round(a2, 3)) + " * x2 + " + str(
    round(a3, 3)) + " * x3")
