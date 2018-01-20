from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Среднее значение
mean = x_train.mean(axis=0)
# Стандартное отклонение
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

mse, mae = model.evaluate(x_test, y_test, verbose=0)

print("Средняя абсолютная ошибка (тысяч долларов):", mae)

pred = model.predict(x_test)

print("Предсказанная стоимость:", pred[9][0], ", правильная стоимость:", y_test[9])


print("Сохраняем сеть")
# Сохраняем сеть для последующего использования
# Генерируем описание модели в формате json
model_json = model.to_json()
json_file = open("mnist_model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
# Записываем данные о весах в файл
model.save_weights("mnist_model.h5")
print("Сохранение сети завершено")