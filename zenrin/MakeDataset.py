import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from method import *
from method2d import *
import figure2d as F
import figure2 as F2
from Projection import Plane3DProjection

import open3d

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

def CheckPolygonInternal(vertices, AABB):
    xmin, xmax, ymin, ymax = AABB

    X, Y = Disassemble2d(vertices)

    # 全ての頂点がAABB内にあればTrue
    if np.all((xmin <= X) & (X <= xmax) & (ymin <= Y) & (Y <= ymax)):
        return True

    return False


def ConstructAABBObject2d(max_p, min_p):
    # 図形作成処理
    lx1 = F.line([1, 0, max_p[0]])
    lx2 = F.line([-1, 0, -min_p[0]])
    ly1 = F.line([0, 1, max_p[1]])
    ly2 = F.line([0, -1, -min_p[1]])

    AABB = F.inter(lx1, F.inter(lx2, F.inter(ly1, ly2)))

    return AABB

# 図形の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 図形の点群+ノイズの合計値はNとし、図形点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に図形が入っていなかったら再生成
def MakePointSet(fig_type, N, rate=Random(0.5, 1),  low=-100, high=100, grid_step=50):
    # 平面点群の割合をランダムで決める
    #rate = Random(0.5, 1)
    size = int(N*rate//1)
    #print(size)

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
        if fig_type != 0:
            # 円以外なら、頂点の内外判定でOK
            vertices = fig.CalcVertices()
            if CheckPolygonInternal(vertices, AABB):
                break

        else:
            # 円なら上下左右の4点をとり、内外判定
            u, v, r = fig.p
            vertices = np.array([[u+r, v], [u-r, v], [u, v+r], [u, v-r]])
            if CheckPolygonInternal(vertices, AABB):
                break

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
# 1. 図形の点群+ノイズの合計値はN, rateによって図形点群の割合が変わる
# 2. AABBは射影した図形点群のAABBの1~1.5割増に作成
def MakePointSet3D(fig_type, N, rate=Random(0.5, 1), low=-100, high=100, grid_step=50):

    print("rate:{}".format(rate))
    size = int(N*rate//1)
    print("size:{}".format(size))

    ## 平面図形設定 + 点群作成 ##
    fig2d, points2d, _ = MakePointSet(fig_type, size, rate=1.0)

    ## 平面ランダム生成 ##
    plane = RandomPlane()

    ## 平面に2D点群を射影 ##
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
    # 平面3d射影
    center, points3d = Plane3DProjection(points2d, fig2d.p, u, v, O)

    max_p, min_p, l = buildAABB(points3d)

    ## ノイズ用AABBを、図形点群のAABBの1~1.5倍増しに作成 ##
    xmax, ymax, zmax = max_p
    xmin, ymin, zmin = min_p
    AABB = [xmin, xmax, ymin, ymax, zmax, zmin]

    p = [Random(1, 1.5) for i in range(6)]

    xmax = xmax + (xmax - xmin)/2 * p[0]
    xmin = xmin - (xmax - xmin)/2 * p[1]
    ymax = ymax + (ymax - ymin)/2 * p[2]
    ymin = ymin - (ymax - ymin)/2 * p[3]
    zmax = zmax + (zmax - zmin)/2 * p[4]
    zmin = zmin - (zmax - zmin)/2 * p[5]

    max_p = np.array([xmax, ymax, zmax])
    min_p = np.array([xmin, ymin, zmin])

    ## 点群を法線方向に"微量な値"分動かす ##
    # AABBの対角線の長さlを"微量な値"に利用
    tn = np.array([Random(0, 0.01)*l*n for i in range(points3d.shape[0])])
    points3d += tn
    # print("l:{}".format(l))

    ## ノイズ生成、図形点群と結合 ##
    if N == size:
        # ノイズなし
        points = points3d[:, :]

    else:
        # AABBにランダムにノイズ生成
        noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(N-size)])
        # 平面点群とノイズの結合 
        points = np.concatenate([points3d, noise])

    # 図形点群は1, ノイズは0でラベル付け
    trueIndex = np.array([True if i < size else False for i in range(points.shape[0])])

    # シャッフルしておく
    perm = np.random.permutation(points.shape[0])
    points = points[perm]
    trueIndex = trueIndex[perm]

    # #点群をnp配列⇒open3d形式に
    # pointcloud = open3d.PointCloud()
    # pointcloud.points = open3d.Vector3dVector(points)

    # # color
    # colors3 = np.asarray(pointcloud.colors)
    # colors3 = np.array([[255, 130, 0] if trueIndex[i] else [0, 0, 255] for i in range(points.shape[0])]) / 255
    # pointcloud.colors = open3d.Vector3dVector(colors3)

    # # 法線推定
    # open3d.estimate_normals(
    # pointcloud,
    # search_param = open3d.KDTreeSearchParamHybrid(
    # radius = l*0.05, max_nn = 100))

    # # 法線の方向を視点ベースでそろえる
    # open3d.orient_normals_towards_camera_location(
    # pointcloud,
    # camera_location = np.array([0., 10., 10.], 
    # dtype="float64"))

    # #nキーで法線表示
    # open3d.draw_geometries([pointcloud])

    return center, fig2d.p, plane.p, points, AABB, trueIndex

# 図形の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 図形の点群+ノイズの合計値はNとし、図形点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に図形が入っていなかったら再生成
def MakeSign2D(fig_type, N, rate=Random(0.5, 1),  low=-100, high=100, grid_step=50):
    # 平面点群の割合をランダムで決める
    #rate = Random(0.5, 1)
    size = int(N*rate//1)
    #print(size)

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
        if fig_type != 0:
            # 円以外なら、頂点の内外判定でOK
            vertices = fig.CalcVertices()
            if CheckPolygonInternal(vertices, AABB):
                break

        else:
            # 円なら上下左右の4点をとり、内外判定
            u, v, r = fig.p
            vertices = np.array([[u+r, v], [u-r, v], [u, v+r], [u, v-r]])
            if CheckPolygonInternal(vertices, AABB):
                break

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

# 図形の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 図形の点群+ノイズの合計値はN, rateによって図形点群の割合が変わる
# 2. AABBは射影した図形点群のAABBの1~1.5割増に作成
def MakeSign3D(fig_type, N, rate=Random(0.5, 1), low=-100, high=100, grid_step=50):

    print("rate:{}".format(rate))
    size = int(N*rate//1)
    print("size:{}".format(size))

    ## 平面図形設定 + 点群作成 ##
    fig2d, points2d, _ = MakePointSet(fig_type, size, rate=1.0)

    ## 平面ランダム生成 ##
    plane = RandomPlane()

    ## 平面に2D点群を射影 ##
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
    # 平面3d射影
    para3d, points3d = Plane3DProjection(points2d, fig2d, u, v, O)

    max_p, min_p, l = buildAABB(points3d)

    ## ノイズ用AABBを、図形点群のAABBの1~1.5倍増しに作成 ##
    xmax, ymax, zmax = max_p
    xmin, ymin, zmin = min_p
    AABB = [xmin, xmax, ymin, ymax, zmax, zmin]

    p = [Random(1, 1.5) for i in range(6)]

    xmax = xmax + (xmax - xmin)/2 * p[0]
    xmin = xmin - (xmax - xmin)/2 * p[1]
    ymax = ymax + (ymax - ymin)/2 * p[2]
    ymin = ymin - (ymax - ymin)/2 * p[3]
    zmax = zmax + (zmax - zmin)/2 * p[4]
    zmin = zmin - (zmax - zmin)/2 * p[5]

    max_p = np.array([xmax, ymax, zmax])
    min_p = np.array([xmin, ymin, zmin])

    ## 点群を法線方向に"微量な値"分動かす ##
    # AABBの対角線の長さlを"微量な値"に利用
    tn = np.array([Random(0, 0.01)*l*n for i in range(points3d.shape[0])])
    points3d += tn
    # print("l:{}".format(l))

    ## ノイズ生成、図形点群と結合 ##
    if N == size:
        # ノイズなし
        points = points3d[:, :]

    else:
        # AABBにランダムにノイズ生成
        noise = np.array([[Random(xmin, xmax), Random(ymin, ymax), Random(zmin, zmax)] for i in range(N-size)])
        # 平面点群とノイズの結合 
        points = np.concatenate([points3d, noise])

    # 図形点群は1, ノイズは0でラベル付け
    trueIndex = np.array([True if i < size else False for i in range(points.shape[0])])

    # シャッフルしておく
    perm = np.random.permutation(points.shape[0])
    points = points[perm]
    trueIndex = trueIndex[perm]

    # #点群をnp配列⇒open3d形式に
    # pointcloud = open3d.PointCloud()
    # pointcloud.points = open3d.Vector3dVector(points)

    # # color
    # colors3 = np.asarray(pointcloud.colors)
    # colors3 = np.array([[255, 130, 0] if trueIndex[i] else [0, 0, 255] for i in range(points.shape[0])]) / 255
    # pointcloud.colors = open3d.Vector3dVector(colors3)

    # # 法線推定
    # open3d.estimate_normals(
    # pointcloud,
    # search_param = open3d.KDTreeSearchParamHybrid(
    # radius = l*0.05, max_nn = 100))

    # # 法線の方向を視点ベースでそろえる
    # open3d.orient_normals_towards_camera_location(
    # pointcloud,
    # camera_location = np.array([0., 10., 10.], 
    # dtype="float64"))

    # #nキーで法線表示
    # open3d.draw_geometries([pointcloud])

    return para3d, fig2d.p, plane.p, points, AABB, trueIndex

#MakePointSet3D(1, 1000, rate=0.5)