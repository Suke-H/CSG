from keras.models import Sequential
from keras.layers import Dense

import random
import numpy as np
import matplotlib.pyplot as plt

#目標の関数
f = lambda x: np.sin(x) +  0.3*x**2

#nの数だけランダムにy=sinxを生成
def get_data(n):
    x = np.random.uniform(-10, 10, n)
    x = np.reshape(x, [n,1])
    y = f(x)

    return x, y

#
batch = 100
epoch = 100
test_size = 10000

#訓練データ作成
x_train, t_train = get_data(batch * epoch)
print("x_train:{}, t_train:{}".format(x_train.shape, t_train.shape))

#モデル作成
model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(1, )))
model.add(Dense(units=1))
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.summary()

#訓練
model.fit(x_train, t_train,
            batch_size=batch,
            epochs=epoch,
            verbose=1)



#検証
#score = model.evaluate(x_test, t_test)
#print(score)

#予測
x_test = np.linspace(-10, 10, test_size)
x_test = np.reshape(x_test, [test_size,1])
t_test = f(x_test)
plt.plot(x_test, t_test, color="red")

#print(x_train.shape, x_test.shape)

y_test = model.predict(x_test)

#y_test = np.asarray([])
#for i in range(100):
#    y_test = np.append(y_test, model.evaluate(x_test[i], batch_size=100))

plt.plot(x_test, y_test, color="blue")
plt.show()







