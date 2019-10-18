from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
import queue
import collections

import figure2 as F
from PreProcess2 import PreProcess2
from method import *

def CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03, alpha=np.pi/12):

    #条件を満たす点群を取り出す
    marked_index, _, _, _, _ = MarkPoints(figure, X, Y, Z, normals, epsilon, alpha)

    #マークされてないときの処理
    if marked_index == []:
        return [0], [0], [0], 0, np.array([0])

    #ラベル化をする(label_list[i]にはi番目の点のラベル情報が入っている。0は無し)
    label_list = LabelPoints(points, marked_index, k=5)

    #ラベルの種類の数
    label_num = np.max(label_list)

    #各ラベルの数を辞書型で保存
    label_dict = collections.Counter(label_list)
    val = list(label_dict.values())[1:]

    #一番多いラベルを見つける
    max_label = list(label_dict.keys())[val.index(max(val))+1]

    #最大ラベルのインデックス、点群、その数
    max_label_index = np.where(label_list == max_label)
    max_label_points = points[max_label_index]
    max_label_num = val[val.index(max(val))] 

    X, Y, Z = Disassemble(max_label_points)

    print("label_num:{}\nval:{}\nmax_label:{}".format(label_num, val, max_label_num))

    return X, Y, Z, max_label_num, max_label_index[0]
    
def MarkPoints(figure, X, Y, Z, normals, epsilon, alpha):
    #|f(x,y,z)|<εを満たす点群だけにする
    D = figure.f_rep(X, Y, Z)
    index_1 = np.where(np.abs(D)<epsilon)
    #print("f<ε：{}".format(len(index_1[0])))

    #次にcos-1(|nf*ni|)<αを満たす点群だけにする
    T = np.arccos(np.abs(np.sum(figure.normal(X,Y,Z) * normals, axis=1)))
    # (0<T<pi/2のはずだが念のため絶対値をつけてる)
    index_2 = np.where(np.abs(T)<alpha)
    #print("θ<α：{}".format(len(index_2[0])))

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

"""
#グラフ作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#事前準備
#figure = F.plane([0,0,1,0])
figure = F.sphere([0.75, 0.75, 0.75, 0.75])
points, X, Y, Z, normals, length = PreProcess2()

#points
ax.plot(X, Y, Z, marker="o",linestyle="None",color="white")

#図形plot
plot_implicit(ax, figure.f_rep, points, 1, 100)

########
X, Y, Z, num = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.04*length, alpha=np.pi/12)

print("num{}".format(num))
ax.plot(X, Y, Z, marker=".",linestyle="None",color="red")

plt.show()
"""