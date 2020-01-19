import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from method import *

from method2d import *
import figure2d as F

import figure2 as F2

def RandomCircle(x_min=-100, x_max=100, y_min=-100, y_max=100, r_min=0, r_max=20):
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    r = Random(r_min, r_max)

    return F.circle([x, y, r])

def RandomTriangle(x_min=-100, x_max=100, y_min=-100, y_max=100, r_min=0, r_max=20):
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    r = Random(r_min, r_max)
    t = Random(0, np.pi*2/3)

    return F.tri([x, y, r, t])

def RandomRectangle(x_min=-100, x_max=100, y_min=-100, y_max=100, w_min=0, w_max=20, h_min=0, h_max=20):
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    w = Random(w_min, w_max)
    h = Random(h_min, h_max)
    t = Random(0, np.pi*2/3)

    return F.rect([x, y, w, h, t])

def RandomPlane(high=1000):
    # 法線作成
    n = np.array([0, 0, 0])
    while LA.norm(n) == 0:
        n = np.random.rand(3)

    n = n / LA.norm(n)
    a, b, c = n

    # d作成
    d = Random(-high, high)

    return F2.plane([a, b, c, d])

def CheckInternal(fig_type, figure, AABB):
    xmin, xmax, ymin, ymax = AABB

    # 円なら
    if fig_type == 0:
        x, y, r = figure.p
        ## 描画
        sign = MakePoints2d(figure.f_rep, AABB=AABB ,grid_step=100, epsilon=0.01, down_rate = 0.5)
        signX, signY = Disassemble2d(sign)
        plt.plot(signX, signY, marker=".",linestyle="None",color="orange")
        max_p = [AABB[1], AABB[3]]
        min_p = [AABB[0], AABB[2]]
        AABBViewer2d(max_p, min_p)

        plt.show()

        if (abs(xmax-x) > r) and (abs(xmin-x) > r) and (abs(ymax-y) > r) and (abs(ymax-y) > r):
            return True

        else:
            return False

    # 多角形なら頂点を取得する
    vertices = figure.CalcVertices()
    X, Y = Disassemble2d(vertices)

    ## 描画
    #plt.plot(X, Y, marker="o",linestyle="None",color="red")
    # sign = MakePoints(figure.f_rep, AABB=AABB ,grid_step=100, epsilon=0.01, down_rate = 0.5)
    # signX, signY = Disassemble2d(sign)
    # plt.plot(signX, signY, marker=".",linestyle="None",color="orange")
    # max_p = [AABB[1], AABB[3]]
    # min_p = [AABB[0], AABB[2]]
    # AABBViewer(max_p, min_p)

    # plt.show()

    # 全ての頂点がAABB内にあればTrue
    if np.all((xmin <= X) & (X <= xmax)) and np.all((ymin <= Y) & (Y <= ymax)):
        return True

    return False

# 図形の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 図形の点群+ノイズの合計値はNとし、図形点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に図形が入っていなかったら再生成
def MakePointSet(fig_type, N, rate=Random(0.5, 1),  low=-100, high=100, grid_step=50):
    # 平面点群の割合をランダムで決める
    #rate = Random(0.5, 1)
    size = int(N*rate//1)
    print(size)

    # AABBランダム生成
    while True:
        AABB = []
        for i in range(2):
            x1 = Random(low, high)
            x2 = Random(low, high)
            if x1>=x2:
                x_axis = [x2, x1]
            else:
                x_axis = [x1, x2]
            AABB.extend(x_axis)

        #print(AABB)
        xmin, xmax, ymin, ymax = AABB
        w = abs(xmax-xmin)
        h = abs(ymax-ymin)

        # 縦横比が8割以下ならやり直し
        # (横が大きいなら縦 >= 0.8*横)
        if (w >= h and h >= 0.8*w) or (h >= w and w >= 0.8*h):
            break

    # 半径の生成条件に対角線の長さを利用する    
    #l = w if w <= h else h
    l = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)

    while True:

        # 図形ランダム生成
        if fig_type == 0:
            fig = RandomCircle(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, r_min=l/10, r_max=l/2)
        elif fig_type == 1:
            fig = RandomTriangle(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, r_min=l/10, r_max=l/2)
        elif fig_type == 2:
            fig = RandomRectangle(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, w_min=w/8, w_max=w/1.5, h_min=h/8, h_max=h/1.5)

        # AABB内に図形がなければ再生成
        if CheckInternal(fig_type, fig, AABB):
            break

    # 平面点群を生成
    fig_points = InteriorPoints(fig.f_rep, AABB, size, grid_step=grid_step)

    # N-size点のノイズ生成
    xmin, xmax, ymin, ymax = AABB

    # ノイズなし
    if N == size:
        return fig, fig_points, AABB

    noise = np.array([[Random(xmin, xmax), Random(ymin, ymax)] for i in range(N-size)])

    #print(plane_points.shape, noise.shape)

    # 平面点群とノイズの結合
    # シャッフルもしておく
    points = np.concatenate([fig_points, noise])
    np.random.shuffle(points)

    return fig, points, AABB

def ConstructAABBObject(max_p, min_p):
    px1 = F2.plane([1, 0, 0, max_p[0]])
    px2 = F2.plane([-1, 0, 0, -min_p[0]])
    py1 = F2.plane([0, 1, 0, max_p[1]])
    py2 = F2.plane([0, -1, 0, -min_p[1]])
    pz1 = F2.plane([0, 0, 1, max_p[1]])
    pz2 = F2.plane([0, 0, -1, -min_p[1]])

    AABB = F2.AND(F2.AND(F2.AND(F2.AND(F2.AND(px1, px2), py1), py2), pz1), pz2)

    return AABB


# 図形の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 図形の点群+ノイズの合計値はNとし、図形点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に図形が入っていなかったらAABB再生成
def MakePointSet3D(fig_type, N, rate=Random(0.5, 1), low=-100, high=100, grid_step=50):
    # 平面点群の割合をランダムで決める
    #rate = Random(0.5, 1)
    print("rate:{}".format(rate))
    size = int(N*rate//1)
    print(size)


    # 平面図形設定 + 点群作成
    fig, points2d, _ = MakePointSet(fig_type, N, rate=1.0)

    # 平面ランダム生成
    plane = RandomPlane()

    # 平面上の2点をランダムに定める
    a, b, c, d = plane.p
    ox, oy, ax, ay = Random(low, high), Random(low, high), Random(low, high), Random(low, high)
    oz = (d - a*ox - b*oy) / c
    az = (d - a*ax - b*ay) / c
    O = np.array([ox, oy, oz])
    A = np.array([ax, ay, az])

    # uを定める
    u = norm(A - O)

    # 平面のnよりv算出
    n = np.array([a, b, c])
    v = norm(np.cross(u, n))

    # 三次元に射影
    uv = np.array([u, v])
    points3d = np.dot(points2d, uv) + np.array([O for i in range(points2d.shape[0])])

    # 点群を法線方向に微量動かす
    max_p, min_p = buildAABB(points3d)
    print((max_p[0]-min_p[0]), (max_p[1]-min_p[1]), (max_p[2]-min_p[2]))
    l = np.sqrt((max_p[0]-min_p[0])**2 + (max_p[1]-min_p[1])**2 + (max_p[2]-min_p[2])**2)
    print(l)
    tn = np.array([Random(0, 1)*n for i in range(points3d.shape[0])])
    points3d += tn

    ###################################################

    # AABBランダム生成
    # while True:
    #     while True:
    #         max_p = []
    #         min_p = []
    #         for i in range(3):
    #             x1 = Random(low, high)
    #             x2 = Random(low, high)
    #             if x1>=x2:
    #                 max_p.append(x1)
    #                 min_p.append(x2)
    #             else:
    #                 max_p.append(x2)
    #                 min_p.append(x1)
                
    #     AABB = ConstructAABBObject(max_p, min_p)

    #     X, Y, Z = Disassemble(points3d)
    #     W = AABB.f_rep(X, Y, Z)

    #     # 図形点群がすべてAABB内にあればOK
    #     if np.all(W>0):
    #         continue

    X, Y, Z = Disassemble(points3d)

    xmax, ymax, zmax = max_p
    xmin, ymin, zmin = min_p

    p = [Random(1, 1.5) for i in range(6)]

    xmax = xmax + (xmax - xmin)/2 * p[0]
    xmin = xmin - (xmax - xmin)/2 * p[1]
    ymax = ymax + (ymax - ymin)/2 * p[2]
    ymin = ymin - (ymax - ymin)/2 * p[3]
    zmax = zmax + (zmax - zmin)/2 * p[4]
    zmin = zmin - (zmax - zmin)/2 * p[5]

    #AABB = [xmin, xmax, ymin, ymax, zmin, zmax]
    max_p = np.array([xmax, ymax, zmax])
    min_p = np.array([xmin, ymin, zmin])

    print(max_p, min_p)

    # ノイズなし
    # if N == size:
    #     return fig, fig_points, AABB

    noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(N-size)])

    # 平面点群とノイズの結合
    # シャッフルもしておく
    points = np.concatenate([points3d, noise])
    np.random.shuffle(points)

    #グラフの枠を作っていく
    fig_plt = plt.figure()
    ax = Axes3D(fig_plt)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    X, Y, Z = Disassemble(points)

    ax.plot(X, Y, Z, marker="o", linestyle='None', color="red")

    AABBViewer(ax, max_p, min_p)
    plt.show()

MakePointSet3D(2, 500)
#点群,法線, 取得
# max_p = [1, 1, 1]
# min_p = [-1, -1, -1]
# AABB = ConstructAABBObject(max_p, min_p)

# points = MakePoints(AABB.f_rep, bbox=(-1.2, 1.2), grid_step=30)

# print(points.shape)
# #グラフの枠を作っていく
# fig_plt = plt.figure()
# ax = Axes3D(fig_plt)

# #軸にラベルを付けたいときは書く
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# X, Y, Z = Disassemble(points)
# ax.plot(X, Y, Z, marker=".", linestyle='None', color="red")

# plot_implicit(ax, AABB.f_rep, points, AABB_size=1.5, contourNum=100)
# plt.show()