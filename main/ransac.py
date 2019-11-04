import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from PreProcess2 import PreProcess2
from method import *
import figure2 as F
from FigureDetection import CountPoints

def PlaneDict(points, normals, X, Y, Z, length):
    n = points.shape[0]
    # ランダムに3点ずつ抽出
    # 今のままだとチョイスされた点はどの組にもかぶらないようにされてるため、nC3となるようにチョイスしたい
    points_set = points[np.random.choice(n, size=(int((n-n%3)/3), 3), replace=False), :]

    # 分割
    # [a1, b1, c1] -> [a1] [b1, c1]
    a0, a1 = np.split(points_set, [1], axis=1)

    # a2 = [[b1-a1], ...,[bn-an]]
    #      [[c1-a1], ...,[cn-an]]
    a2 = np.transpose(a1-a0, (1,0,2))

    # n = (b-a) × (c-a)
    n = np.cross(a2[0], a2[1])

    # 単位ベクトルに変換
    n = norm(n)

    # d = n・a
    a0 = np.reshape(a0, (a0.shape[0],3))
    d = np.sum(n*a0, axis=1)

    # パラメータ
    # p = [nx, ny, nz, d]
    d = np.reshape(d, (d.shape[0],1))
    p = np.concatenate([n, d], axis=1)

    # 平面生成
    Planes = [F.plane(p[i]) for i in range(p.shape[0])]

    # フィットしている点の数を数える
    Scores = [CountPoints(Planes[i], points, X, Y, Z, normals, epsilon=0.08*length, alpha=np.pi/12)[3] for i in range(p.shape[0])]

    print(Scores, p[Scores.index(max(Scores))])

    return Planes[Scores.index(max(Scores))]

def SphereDict(points, normals, X, Y, Z, length):
    n = points.shape[0]
    # ランダムに3点ずつ抽出
    # 今のままだとチョイスされた点はどの組にもかぶらないようにされてるため、nC3となるようにチョイスしたい
    index = np.random.choice(n, size=(int((n-n%2)/2), 2), replace=False)
    points_set = points[index, :]
    normals_set = normals[index, :]

    num = points_set.shape[0]

    radius = lambda p1, p2, n1, n2 : np.dot(p1-p2, n1-n2) / np.linalg.norm(n1-n2)**2
    center = lambda p1, n1, r : p1 - r * n1

    r = [radius(points_set[i][0], points_set[i][1], normals_set[i][0], normals_set[i][1]) for i in range(num)]
    c = [center(points_set[i][0], normals_set[i][0], r[i]) for i in  range(num)]

    print(np.array(r).shape, np.array(c).shape)

    # パラメータ
    # p = [x0, y0, z0, r]
    r = np.reshape(r, (num,1))
    p = np.concatenate([c, r], axis=1)

    # 球面生成
    Spheres = [F.sphere(p[i]) for i in range(num)]

    # フィットしている点の数を数える
    Scores = [CountPoints(Spheres[i], points, X, Y, Z, normals, epsilon=0.01*length, alpha=np.pi/12)[3] for i in range(num)]

    print(Scores, max(Scores))
    print(p[Scores.index(max(Scores))])

    return Spheres[Scores.index(max(Scores))]
    

points, X, Y, Z, normals, length = PreProcess2()

#figure = PlaneDict(points, normals, X, Y, Z, length)
figure = SphereDict(points, normals, X, Y, Z, length)

#グラフ作成
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#points
ax.plot(X, Y, Z, marker="o",linestyle="None",color="white")

#図形plot
plot_implicit(ax, figure.f_rep, points, 1, 100)

plt.show()
