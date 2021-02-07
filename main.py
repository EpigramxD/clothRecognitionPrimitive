import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
# подключаем модель нейросети, в которой слои идут друг за здургом
from tensorflow.keras.models import Sequential
# подключаем модель полносвязного слоя
from tensorflow.keras.layers import Dense
# подключаем утилиты
from tensorflow.keras import utils

from tensorflow.keras.preprocessing import image


def show_images(images, categories_values, categories_names):
    for i in range(100, 150):
        plt.subplot(5, 10, i - 100 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(categories_names[categories_values[i]])
    plt.show()


# названия классов
classes = ['футболка', 'брюки', 'свитер', 'платье',
           'пальто', 'туфли', 'рубашка',
           'кроссовки', 'сумка', 'ботинки']


# загружаем данные
# x_train - изображения для обучения
# y_train - что изображено на пикче
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

show_images(x_train, y_train, classes)

# преобразование пикч в вектора
# 60000 пикч преобразуется в вектор
# 784 пиксела в каждой пикче (28 * 28)
x_train = x_train.reshape(60000, 784)
# нормализация данных (данные будут от 0 до 1, это удобно для нейронки)
x_train = x_train / 255
# преобразуем метки в категории
y_train = utils.to_categorical(y_train, 10)
# создаем последовательную сеть
model = Sequential()

# добавляем уровни сети

# входной слой
# функция активации - relu (rectified liner unit)
# если зачение меньше нуля, то выдает ноль
# если значение больше нуля, то возвращает само значение
model.add(Dense(800, input_dim=784, activation="relu"))

# выходной слой
# функция активации - softmax - нормализированная экспоненциальная функция
# используется для представления вероятности
# сумма всех выходных значчений нейронов равна 1
model.add(Dense(10, activation="softmax"))

# компилируем модель
# loss - функция ошибки - категориальная перекрестная интропия
# она очень хороша в задачах классификации, в которых классов больше, чем 2
# можно использовать среднеквадратическую ошибку
# optimizer - тип оптимизатора - стахостически градиентный спуск (SGD)
# metrics - метрика качества, используем accuracy - доля правильных ответов нейронной сети
model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

# выводим модель
print(model.summary())

# обучаем сеть
# передаем данные для обучения и правильные ответы
# batch_size - количество минивыборки
# фактически означает, что после 200 изображений мы будем изменять веса
# то есть сеть будет самокорректироваться после каждых 200 пикч
# epochs - сколько используем эпох
# эпоха - прогон нейронки по всему датасету
# то есть будем 100 раз обучать сеть на одном и том же датасете
# verbose - показывать процесс обучения сети
model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1)

# запускаем сеть на входных данных
predictions = model.predict(x_train)

n = 109
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

# выводим номер класса, который показала нейросеть
print(classes[np.argmax(predictions[n])])

# вывдоим реальное значение
print(classes[np.argmax(y_train[n])])