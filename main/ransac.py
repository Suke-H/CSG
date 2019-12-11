import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from PreProcess2 import PreProcess2
from method import *
import figure2 as F
from FigureDetection import CountPoints
#from FDtest import CountPoints as CPtest
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
    N = 5000
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


def CylinderDict(points, normals, X, Y, Z, length):
    n = points.shape[0]
    N = 5000
    # ランダムに2点ずつN組抽出
    #index = np.array([np.random.choice(n, 2, replace=False) for i in range(N)])
    index = np.random.choice(n, size=(int((n-n%2)/2), 2), replace=False)
    points_set = points[index, :]
    normals_set = normals[index, :]

    num = points_set.shape[0]

    # lambda式が長くなりそうなのでnpメソッドの省略
    N = lambda v: np.linalg.norm(v)
    D = lambda v1, v2: np.dot(v1, v2)

    # 各パラメータの算出式
    radius1 = lambda p1, p2, n1, n2 : (N(n2)**2*D(p1-p2,n1) - D(n1,n2)*D(p1-p2,n2)) / ((N(n1)*N(n2))**2 - D(n1,n2)**2)
    radius2 = lambda p1, p2, n1, n2 : (D(n1,n2)*D(p1-p2,n1) - N(n1)**2*D(p1-p2,n2)) / ((N(n1)*N(n2))**2 - D(n1,n2)**2)
    point1 = lambda p1, n1, r1: p1 - r1*n1
    point2 = lambda p2, n2, r2: p2 - r2*n2
    #direction = lambda q1, q2: norm(q2-q1)
    truth_radius = lambda p1, q1: N(p1 - q1)

    # q1, q2:方向ベクトルの2点
    # w:方向ベクトル
    # R:半径
    r1 = [radius1(points_set[i][0], points_set[i][1], normals_set[i][0], normals_set[i][1]) for i in range(num)]
    r2 = [radius2(points_set[i][0], points_set[i][1], normals_set[i][0], normals_set[i][1]) for i in range(num)]
    q1 = [point1(points_set[i][0], normals_set[i][0], r1[i]) for i in range(num)]
    q2 = [point2(points_set[i][1], normals_set[i][1], r2[i]) for i in range(num)]
    # wは正規化
    w = [norm(q2[i]-q1[i]) for i in range(num)]
    R = [truth_radius(points_set[i][0], q1[i]) for i in range(num)]

    print(num)

    ### R < lengthの条件を満たさないものを削除 ###
    index = np.where(R >= length)
    R = np.delete(R, index)
    q1 = np.delete(q1, index, axis=0)
    w = np.delete(w, index, axis=0)

    num = len(R)
    print(num, q1.shape, w.shape)

    # パラメータ
    # p = [x0, y0, z0, a, b, c, r]
    R = np.reshape(R, (num,1))
    p = np.concatenate([q1, w, R], axis=1)

    # 球面生成
    Cylinders = [F.cylinder(p[i]) for i in range(num)]

    # フィットしている点の数を数える
    Scores = [CountPoints(Cylinders[i], points, X, Y, Z, normals, epsilon=0.01*length, alpha=np.pi/10)[3] for i in range(num)]

    print(p[Scores.index(max(Scores))])

    return p[Scores.index(max(Scores))], Cylinders[Scores.index(max(Scores))]

def ConeDict(points, normals, X, Y, Z, length):
    n = points.shape[0]
    N = 5000
    # ランダムに3点ずつN組抽出
    index = np.array([np.random.choice(n, 3, replace=False) for i in range(N)])
    #index = np.random.choice(n, size=(int((n-n%3)/3), 3), replace=False)
    points_set = points[index, :]
    normals_set = normals[index, :]

    num = points_set.shape[0]

    # 省略
    DET = lambda v1, v2, v3: np.linalg.det(np.stack([v1, v2, v3]))
    DOT = lambda v1, v2: np.dot(v1, v2)


    # 各パラメータの算出式
    """
    det_A = lambda n1, n2, n3: np.linalg.det(np.stack([n1, n2, n3]))
    det_A1 = lambda p1, n2, n3: np.linalg.det(np.stack([p1, n2, n3]))
    det_A2 = lambda p2, n1, n3: np.linalg.det(np.stack([n1, p2, n3]))
    det_A3 = lambda p3, n1, n2: np.linalg.det(np.stack([n1, n2, p3]))
    apex = lambda A, A1, A2, A3 : np.array([A1/A, A2/A, A3/A])
    """
    d_list = lambda p1, p2, p3, n1, n2, n3: np.array([DOT(n1,p1), DOT(n2,p2), DOT(n3,p3)])

    apex = lambda p1, p2, p3, n1, n2, n3: \
        np.array([DET(d_list(p1,p2,p3,n1,n2,n3), n2, n3) / DET(n1, n2, n3),\
                DET(n1, d_list(p1,p2,p3,n1,n2,n3), n3) / DET(n1, n2, n3), \
                DET(n1, n2, d_list(p1,p2,p3,n1,n2,n3)) / DET(n1, n2, n3)])

    """
    point = lambda p1, p2, p3, c: np.array([c+norm(p1-c), c+norm(p2-c), c+norm(p2-c)])
    normal = lambda a1, a2, a3: np.cross(a2-a1, a3-a1)
    """

    # 平面の法線(=direction)の向きをaがない半空間の方向にしたいので、
    # f(a)>0のときnormal, f(a)<0のとき-normalを返す
    # (f=d-(ax+by+cz)だとf(a)>0のときnはaがない方向、つまりnは内部(領域)から発散する方向に向いている)

    #direction = lambda p1, p2, p3, a: norm(np.cross(norm(p2-a)-norm(p1-a), norm(p3-a)-norm(p1-a)))

    normal = lambda p1, p2, p3, a: norm(np.cross(norm(p2-a)-norm(p1-a), norm(p3-a)-norm(p1-a)))
    plane_frep = lambda p1, p2, p3, a: lambda x: DOT(normal(p1,p2,p3,a), a+norm(p1-a)) - DOT(normal(p1,p2,p3,a), x)
    direction = lambda p1, p2, p3, a: normal(p1,p2,p3,a) if plane_frep(p1,p2,p3,a)(a) > 0 else -normal(p1,p2,p3,a)
    
    theta = lambda p1, a, w: np.arccos(np.dot(norm(p1-a), w))
    #theta2 = lambda p2, a, w: np.arccos(np.dot(norm(p2-a), w))
    #theta3 = lambda p3, a, w: np.arccos(np.dot(norm(p3-a), w))

    # q1, q2:方向ベクトルの2点
    # w:方向ベクトル
    # R:半径
    a = np.array([apex(points_set[i][0], points_set[i][1], points_set[i][2], normals_set[i][0], normals_set[i][1], normals_set[i][2]) for i in range(num)])
    w = np.array([direction(points_set[i][0], points_set[i][1], points_set[i][2], a[i]) for i in range(num)])
    t = np.array([theta(points_set[i][0], a[i], w[i]) for i in range(num)])
    #t2 = [theta2(points_set[i][1], a[i], w[i]) for i in range(num)]
    #t3 = [theta3(points_set[i][2], a[i], w[i]) for i in range(num)]
    #t = np.array([(t[i]+t2[i]+t3[i])/3 for i in range(num)])

    print(w[:5])
    print(t[:5])

    print(num)

    ### 10 < theta < 60の条件を満たさないものを削除 ###
    index = np.where((t < np.pi/(180/10)) | (t > np.pi/(180/60)))
    t = np.delete(t, index)
    a = np.delete(a, index, axis=0)
    w = np.delete(w, index, axis=0)

    num = len(t)
    print(num)

    # パラメータ
    # p = [x0, y0, z0, a, b, c, theta]
    t = np.reshape(t, (num,1))
    p = np.concatenate([a, w, t], axis=1)

    # 球面生成
    Cones = [F.cone(p[i]) for i in range(num)]

    # フィットしている点の数を数える
    Scores = [CountPoints(Cones[i], points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi/9)[3] for i in range(num)]

    print(p[Scores.index(max(Scores))])

    return p[Scores.index(max(Scores))], Cones[Scores.index(max(Scores))]


def RANSAC(fig, points, normals, X, Y, Z, length):
    # 図形に応じてRANSAC
    if fig==0:
        res1, figure = SphereDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.01*length, np.pi/12

    elif fig==1:
        res1, figure = PlaneDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.08*length, np.pi/9

    elif fig==2:
        res1, figure = CylinderDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.01*length, np.pi/12

    elif fig==3:
        res1, figure = ConeDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.03*length, np.pi/9

    # フィット点を抽出
    MX1, MY1, MZ1, num1, index1 = CountPoints(figure, points, X, Y, Z, normals, epsilon=epsilon, alpha=alpha)

    print("BEFORE_num:{}".format(num1))

    if num1!=0:
        # フィット点を入力にフィッティング処理
        res2 = Fitting(MX1, MY1, MZ1, normals[index1], length, fig, figure.p, epsilon=epsilon, alpha=alpha)
        print(res2.x)

        if fig==0:
            figure = F.sphere(res2.x)

        elif fig==1:
            figure = F.plane(res2.x)

        elif fig==2:
            figure = F.cylinder(res2.x)

        elif fig==3:
            figure = F.cone(res2.x)

        # フィッティング後のスコア出力
        MX2, MY2, MZ2, num2, index2 = CountPoints(figure, points, X, Y, Z, normals, epsilon=epsilon, alpha=alpha)

        print("AFTER_num:{}".format(num2))

        # フィッティング後の方が良ければres2を出力
        if num2 >= num1:
            return res2.x, MX2, MY2, MZ2, num2, index2

    # res1のスコア0 OR res2よりスコアが多い => res1を出力
    return res1, MX1, MY1, MZ1, num1, index1

def RANSAC2(fig, points, normals, X, Y, Z, length):
    # 図形に応じてRANSAC
    if fig==0:
        res1, figure = SphereDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.01*length, np.pi/12

    elif fig==1:
        res1, figure = PlaneDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.08*length, np.pi/9

    elif fig==2:
        res1, figure = CylinderDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.01*length, np.pi/12

    elif fig==3:
        res1, figure = ConeDict(points, normals, X, Y, Z, length)
        epsilon, alpha = 0.03*length, np.pi/9

    # フィット点を抽出
    label_list1, max_label1, num1 = CountPoints(figure, points, X, Y, Z, normals, epsilon=epsilon, alpha=alpha, plotFlag=True)

    print("BEFORE_num:{}".format(len(label_list1)))
    print(list(label_list1)
    """
    if num1!=0:
        # フィット点を入力にフィッティング処理
        res2 = Fitting(MX1, MY1, MZ1, normals[index1], length, fig, figure.p, epsilon=epsilon, alpha=alpha)
        print(res2.x)

        if fig==0:
            figure = F.sphere(res2.x)

        elif fig==1:
            figure = F.plane(res2.x)

        elif fig==2:
            figure = F.cylinder(res2.x)

        elif fig==3:
            figure = F.cone(res2.x)

        # フィッティング後のスコア出力
        label_list2, max_label2, num2 = CountPoints(figure, points, X, Y, Z, normals, epsilon=epsilon, alpha=alpha, plotFlag=True)

        print("AFTER_num:{}".format(num2))

        # フィッティング後の方が良ければres2を出力
        if num2 >= num1:
            return res2.x, label_list2, max_label2, num2
    """
    # res1のスコア0 OR res2よりスコアが多い => res1を出力
    return res1, label_list1, max_label1, num1

"""
points, X, Y, Z, normals, length = PreProcess2()

###
#figure = PlaneDict(points, normals, X, Y, Z, length)
#_, figure = SphereDict(points, normals, X, Y, Z, length)
_, figure =  CylinderDict(points, normals, X, Y, Z, length)

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
#res = Fitting(MX, MY, MZ, normals[index], length, 1, figure.p)
res = Fitting(MX, MY, MZ, normals[index], length, 2, figure.p)
print(res)

### 
#figure = F.sphere(res.x)
#figure = F.plane(res.x)
figure = F.cylinder(res.x)

MX2, MY2, MZ2, max_label_num, _ = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi/12)
print("num2:{}".format(max_label_num))

# フィット点群のplot
ax2.plot(MX2, MY2, MZ2, marker=".",linestyle="None",color="red")

#図形plot
plot_implicit(ax2, figure.f_rep, points, 1, 100)

plt.show()
"""