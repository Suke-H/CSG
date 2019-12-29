import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from method2d import *
import figure2d as F

def RandomTriangle(x_min=-100, x_max=100, y_min=-100, y_max=100, r_min=0, r_max=10):
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    r = Random(r_min, r_max)
    t = Random(0, np.pi*2/3)

    return F.tri([x, y, r, t])

def RandomRectangle(x_min=-100, x_max=100, y_min=-100, y_max=100, w_min=0, w_max=10, h_min=0, h_max=10):
    x = Random(x_min, x_max)
    y = Random(y_min, y_max)
    w = Random(w_min, w_max)
    h = Random(h_min, h_max)
    t = Random(0, np.pi*2/3)

    return F.rect([x, y, w, h, t])

def CheckInternal(figure, AABB):
    xmin, xmax, ymin, ymax = AABB

    # 多角形なら頂点を取得する
    vertices = figure.CalcVertices()
    X, Y = Disassemble2d(vertices)
    plt.plot(X, Y, marker="o",linestyle="None",color="red")

    print("p:{}".format(figure.p))
    print("vertices:{}".format(vertices))
    
    sign = MakePoints(figure.f_rep, AABB=AABB ,grid_step=100, epsilon=0.01, down_rate = 0.5)
    signX, signY = Disassemble2d(sign)
    plt.plot(signX, signY, marker=".",linestyle="None",color="orange")
    max_p = [AABB[1], AABB[3]]
    min_p = [AABB[0], AABB[2]]
    AABBViewer(max_p, min_p)

    plt.show()

    # 全ての頂点がAABB内にあればTrue
    if np.all((xmin <= X) & (X <= xmax)) and np.all((ymin <= Y) & (Y <= ymax)):
        return True

    return False

# 図形の点群＋ランダムに生成したAABB内にノイズ作成
# <条件>
# 1. 図形の点群+ノイズの合計値はNとし、図形点群の割合(最低0.5以上)をランダムで出す
# 2. AABB内に図形が入っていなかったら再生成
def MakePointSet(N, low=-100, high=100, grid_step=50):
    # 平面点群の割合をランダムで決める
    rate = Random(0.5, 1)
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

        print(AABB)
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
        fig = RandomTriangle(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, r_min=l/10, r_max=l/2)
        #fig = RandomRectangle(x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax, w_min=w/10, w_max=w/1.5, h_min=h/10, h_max=h/1.5)

        # AABB内に図形がなければ再生成
        if CheckInternal(fig, AABB):
            break

    # 平面点群を生成
    fig_points, _, _ = InteriorPoints(fig.f_rep, AABB, size, grid_step=grid_step)

    # N-size点のノイズ生成
    xmin, xmax, ymin, ymax = AABB
    noise = np.array([[Random(xmin, xmax), Random(ymin, ymax)] for i in range(N-size)])

    #print(plane_points.shape, noise.shape)

    # 平面点群とノイズの結合
    # シャッフルもしておく
    points = np.concatenate([fig_points, noise])
    np.random.shuffle(points)

    return fig, points, AABB