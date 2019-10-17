from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import queue

import figure2 as F
from PreProcess2 import PreProcess2
from test_viewer import Disassemble
from method import *



def CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03, alpha=np.pi/12):

    #条件を満たす点群を取り出す
    marked_index, MX, MY, MZ, _ = MarkPoints(figure, X, Y, Z, normals, epsilon, alpha)
    label_list = LabelPoints(points, marked_index, k=5)
    return label_list, MX, MY, MZ
    
def MarkPoints(figure, X, Y, Z, normals, epsilon, alpha):
    #|f(x,y,z)|<εを満たす点群だけにする
    D = figure.f_rep(X, Y, Z)
    index_1 = np.where(np.abs(D)<epsilon)
    print("f<ε：{}".format(len(index_1[0])))

    #次にcos-1(|nf*ni|)<αを満たす点群だけにする
    T = np.arccos(np.abs(np.sum(figure.normal(X,Y,Z) * normals, axis=1)))
    # (0<T<pi/2のはずだが念のため絶対値をつけてる)
    index_2 = np.where(np.abs(T)<alpha)
    print("θ<α：{}".format(len(index_2[0])))

    #どちらも満たすindexを残す
    index = list(filter(lambda x: x in index_2[0], index_1[0]))
    print("f<ε and θ<α：{}".format(len(index)))

    X = X[index]
    Y = Y[index]
    Z = Z[index]

    #points生成
    points = np.stack([X, Y, Z])
    points = points.T

    return list(index), X, Y, Z, points

def LabelPoints(points, marked_index, k=5):
    #pointsの各点に対応するラベルを格納するリスト
    label_list = [0 for i in range(points.shape[0])]
    label = 1

    #キュー作成(先入れ先出し)
    q = queue.Queue()

    for i in range(points.shape[0]):
        #マークされてない or すでにラベルがついてたらスキップ
        if i not in marked_index or label_list[i] != 0:
            continue

        #最初の点にラベルをつける
        label_list[i] = label

        #K近傍の点のindexをキューに格納
        for p in K_neighbor(points, points[i], k):
            q.put(p)

        #キューがなくなるまでデキューし続ける
        while not q.empty():
            x = q.get()

            #マークされた点 and ラベルを付けてない点をデキューした場合
            if x in marked_index and label_list[x] == 0:
                #ラベルを付ける
                label_list[x] = label
                #K近傍をエンキュー
                for p in K_neighbor(points, points[x], k):
                    q.put(p)

        #ラベルを変える
        label = label + 1

    return np.array(label_list)


figure = F.plane([0,0,1,0])
points, X, Y, Z, normals, length = PreProcess2()
label_list , MX, MY, MZ = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.04*length, alpha=np.pi)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#points
ax.plot(X, Y, Z, marker="o",linestyle="None",color="green")
#マーク
ax.plot(MX, MY, MZ, marker=".",linestyle="None",color="orange")

#図形plot
plot_implicit(ax, figure.f_rep, points, 1, 100)

###ラベル化###
max_label = np.max(label_list)
print(max_label)

colorlist = ["r", "y", "b", "c", "m", "g", "k", "w"]

for i in range(1, max_label):
    #同じラベルの点群のみにする
    same_label_points = points[np.where(label_list == i)]

    #plot
    X, Y, Z = Disassemble(same_label_points)
    ax.plot(X, Y, Z, marker=".",linestyle="None",color=colorlist[i%8])


plt.show()
