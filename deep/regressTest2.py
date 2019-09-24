from keras.models import Sequential
from keras.layers import Dense

import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D

#目標の関数(z=x^2+y^2)
f = lambda x: x[0]**2 + x[1]**2

#nの数だけランダムにz=x^2+y^2を生成
def get_data(n):
    #0から1までの格子点
    base = np.random.uniform(0, 1, n)
    xy = np.asarray(list(itertools.product(base, base)))
    #xyz = np.empty((0,3))
    #xyz = np.append(xyz, f(xy), axis=0)
    z = np.empty((0, 1))
    for i in range(n**2):
        z = np.append(z, np.asarray([[f(xy[i])]]), axis=0)

    return xy, z

#試行回数等
batch = 10
epoch = 50
test_size = 100

#訓練データ作成
xy_train, t_train = get_data(batch * epoch)
print(xy_train.shape, t_train.shape)

#モデル作成
model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(2, )))
model.add(Dense(units=1))
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.summary()

#訓練
model.fit(xy_train, t_train,
            batch_size=batch,
            epochs=epoch,
            verbose=1)

#検証
#score = model.evaluate(x_test, t_test)
#print(score)


#予測
base = np.arange(0.0, 1.0, 1.0/test_size)
xy_test = np.asarray(list(itertools.product(base, base)))
t_test = np.asarray([])

for i in range(test_size**2):
    t_test = np.append(t_test, f(xy_test[i]))

z_test = model.predict(xy_test)
z_test = np.reshape(z_test, (test_size**2, ))

print(z_test.shape)

xy_test = xy_test.T
x_test = xy_test[0]
y_test = xy_test[1]

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.plot(x_test,y_test,t_test,marker=".",linestyle='None',color="green")
ax.plot(x_test,y_test,z_test,marker=".",linestyle='None',color="red")

plt.show()







