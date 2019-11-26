import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from PreProcess2 import PreProcess2
from method import *
import figure2 as F
from FigureDetection import CountPoints
from fitting import Fitting

def PlaneDict(points, normals, X, Y, Z, length):
    #print("10000個抽出")
    n = points.shape[0]
    #print(n)
    #N = 5000
    # ランダムに3点ずつN組抽出
    #points_set = points[np.array([np.random.choice(n, 3, replace=False) for i in range(N)]), :]
    points_set = points[np.random.choice(n, size=(int((n-n%3)/3), 3), replace=False), :]
    
    #print("points:{}".format(points_set.shape))

    #print("計算")
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

    #print("平面生成")

    # 平面生成
    Planes = [F.plane(p[i]) for i in range(p.shape[0])]

    #print("点の数を数える")

    # フィットしている点の数を数える
    Scores = [CountPoints(Planes[i], points, X, Y, Z, normals, epsilon=0.08*length, alpha=np.pi/9)[3] for i in range(p.shape[0])]

    print(p[Scores.index(max(Scores))])

    return p[Scores.index(max(Scores))], Planes[Scores.index(max(Scores))]

def SphereDict(points, normals, X, Y, Z, length):
    n = points.shape[0]
    #N = 5000
    # ランダムに2点ずつN組抽出
    #index = np.array([np.random.choice(n, 2, replace=False) for i in range(N)])
    index = np.random.choice(n, size=(int((n-n%2)/2), 2), replace=False)
    points_set = points[index, :]
    normals_set = normals[index, :]

    num = points_set.shape[0]

    # c = p1 - r*n1
    # c = p2 - r*n2 より
    # r = (p1-p2)*(n1-n2)/|n1-n2|^2, c = p1 - r*n1となる
    radius = lambda p1, p2, n1, n2 : np.dot(p1-p2, n1-n2) / np.linalg.norm(n1-n2)**2
    center = lambda p1, n1, r : p1 - r * n1

    # 二点の組[p1, p2], [n1, n2]をradius, centerに代入
    r = [radius(points_set[i][0], points_set[i][1], normals_set[i][0], normals_set[i][1]) for i in range(num)]
    ### r < lengthの条件を満たさないものを除去 ###
    r = [i for i in r if abs(i) <= length]
    print(num)
    num = len(r)
    print(num)
    c = [center(points_set[i][0], normals_set[i][0], r[i]) for i in  range(num)]

    # rはあとで絶対値をつける
    r = list(map(abs, r))

    #print(np.array(r).shape, np.array(c).shape)

    # パラメータ
    # p = [x0, y0, z0, r]
    r = np.reshape(r, (num,1))
    p = np.concatenate([c, r], axis=1)


    # 球面生成
    Spheres = [F.sphere(p[i]) for i in range(num)]

    # フィットしている点の数を数える
    Scores = [CountPoints(Spheres[i], points, X, Y, Z, normals, epsilon=0.01*length, alpha=np.pi/12)[3] for i in range(num)]

    print(p[Scores.index(max(Scores))])

    return p[Scores.index(max(Scores))], Spheres[Scores.index(max(Scores))]

"""
def CylinderDict(points, normals, X, Y, Z, length):
    n = points.shape[0]
    N = 5000
    # ランダムに2点ずつN組抽出
    index = np.array([np.random.choice(n, 2, replace=False) for i in range(N)])
    #index = np.random.choice(n, size=(int((n-n%2)/2), 2), replace=False)
    points_set = points[index, :]
    normals_set = normals[index, :]

    num = points_set.shape[0]

    radius1 = lambda p1, p2, n1, n2 : np.dot(p1-p2, n1-n2) / np.linalg.norm(n1-n2)**2

    r = [radius(points_set[i][0], points_set[i][1], normals_set[i][0], normals_set[i][1]) for i in range(num)]
    c = [center(points_set[i][0], normals_set[i][0], r[i]) for i in  range(num)]

    #print(np.array(r).shape, np.array(c).shape)

    # パラメータ
    # p = [x0, y0, z0, r]
    r = np.reshape(r, (num,1))
    p = np.concatenate([c, r], axis=1)

    # 球面生成
    Spheres = [F.sphere(p[i]) for i in range(num)]

    # フィットしている点の数を数える
    Scores = [CountPoints(Spheres[i], points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi/9)[3] for i in range(num)]

    print(p[Scores.index(max(Scores))])

    return p[Scores.index(max(Scores))], Spheres[Scores.index(max(Scores))]
"""

def RANSAC(fig, points, normals, X, Y, Z, length):
    # 図形に応じてRANSAC
    if fig==0:
        res1, figure = SphereDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.01*length, np.pi/12

    elif fig==1:
        res1, figure = PlaneDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.08*length, np.pi/9

    # フィット点を抽出
    MX, MY, MZ, num, index = CountPoints(figure, points, X, Y, Z, normals, epsilon=epsilon, alpha=alpha)

    print("BEFORE_num:{}".format(num))

    if num!=0:
        # フィット点を入力にフィッティング処理
        res2 = Fitting(MX, MY, MZ, normals[index], length, fig, figure.p, epsilon=epsilon, alpha=alpha)
        print(res2)

        if fig==0:
            figure = F.sphere(res2.x)

        elif fig==1:
            figure = F.plane(res2.x)

        # フィッティング後のスコア出力
        _, _, _, after_num, _ = CountPoints(figure, points, X, Y, Z, normals, epsilon=epsilon, alpha=alpha)

        print("AFTER_num:{}".format(after_num))

        # フィッティング後の方が良ければres2を出力
        if after_num >= num:
            return res2.x

    # res1のスコア0 OR res2よりスコアが多い => res1を出力
    return res1

"""
points, X, Y, Z, normals, length = PreProcess2()

###
figure = PlaneDict(points, normals, X, Y, Z, length)
#figure = SphereDict(points, normals, X, Y, Z, length)

#グラフ作成
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

#軸にラベルを付けたいときは書く
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

#points
ax1.plot(X, Y, Z, marker="o",linestyle="None",color="white")
ax2.plot(X, Y, Z, marker="o",linestyle="None",color="white")

MX, MY, MZ, max_label_num, index = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi/12)
print("num1:{}".format(max_label_num))

# フィット点群のplot
ax1.plot(MX, MY, MZ, marker=".",linestyle="None",color="red")

#図形plot
plot_implicit(ax1, figure.f_rep, points, 1, 100)

### フィット関数を最適化
#res = Fitting(MX, MY, MZ, normals[index], length, 0, figure.p)
res = Fitting(MX, MY, MZ, normals[index], length, 1, figure.p)
print(res)

### 
#figure = F.sphere(res.x)
figure = F.plane(res.x)

MX2, MY2, MZ2, max_label_num, _ = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi/12)
print("num2:{}".format(max_label_num))

# フィット点群のplot
ax2.plot(MX2, MY2, MZ2, marker=".",linestyle="None",color="red")

#図形plot
plot_implicit(ax2, figure.f_rep, points, 1, 100)

plt.show()
"""