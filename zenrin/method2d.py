import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

#点群データをx, yに分解する

#[x1, y1]         [x1, x2, ..., xn]
#    :       ->   [y1, y2, ..., yn]
#[xn, yn]         
def Disassemble2d(XY):
    X, Y = XY.T[:]

    return X, Y

def line2d(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)

    return x, y
    
def norm(normal):
     #ベクトルが一次元のとき
    if len(normal.shape)==1:
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
            
        else:
            return normal / np.linalg.norm(normal)

    #ベクトルが二次元
    else:
        #各法線のノルムをnormに格納
        norm = np.linalg.norm(normal, ord=2, axis=1)

        #normが0の要素は1にする(normalをnormで割る際に0除算を回避するため)
        norm = np.where(norm==0, 1, norm)

        #normalの各成分をノルムで割る
        norm = np.array([np.full(2, norm[i]) for i in range(len(norm))])
        return normal / norm


# 図形の境界線の点群を生成
def ContourPoints(fn, bbox=(-2.5,2.5), grid_step=50, down_rate = 0.5, epsilon=0.05):
    #import time
    #start = time.time()
    xmin, xmax, ymin, ymax= bbox*2

    #点群X, Y, pointsを作成
    x = np.linspace(xmin, xmax, grid_step)
    y = np.linspace(ymin, ymax, grid_step)

    X, Y= np.meshgrid(x, y)

    # 格子点X, Yをすべてfnにぶち込んでみる
    W = np.array([fn(X[i], Y[i]) for i in range(grid_step)])
    # 変更前
    #W = fn(X, Y)

    #Ｗが0に近いインデックスを取り出す
    index = np.where(np.abs(W)<=epsilon)
    index = [(index[0][i], index[1][i]) for i in range(len(index[0]))]

    #ランダムにダウンサンプリング
    index = random.sample(index, int(len(index)*down_rate//1))

    #格子点から境界面(fn(x,y)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])

    #points作成([[x1,y1],[x2,y2],...])    
    points = np.stack([pointX, pointY])
    points = points.T

    #end = time.time()
    #print("time:{}s".format(end-start))

    return points

# 図形の内部の点群を生成
def InteriorPoints(fn, bbox=(-2.5,2.5), grid_step=50, down_rate = 0.5, epsilon=0.05):
    #import time
    #start = time.time()
    xmin, xmax, ymin, ymax= bbox*2

    #点群X, Y, pointsを作成
    x = np.linspace(xmin, xmax, grid_step)
    y = np.linspace(ymin, ymax, grid_step)

    X, Y= np.meshgrid(x, y)

    # 格子点X, Yをすべてfnにぶち込んでみる
    W = np.array([fn(X[i], Y[i]) for i in range(grid_step)])
    # 変更前
    #W = fn(X, Y)

    #W > 0(=図形の内部)のインデックスを取り出す
    index = np.where(W>0)
    index = [(index[0][i], index[1][i]) for i in range(len(index[0]))]

    #ランダムにダウンサンプリング
    index = random.sample(index, int(len(index)*down_rate//1))

    #格子点から境界面(fn(x,y)=0)に近い要素のインデックスを取り出す
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])

    #points作成([[x1,y1],[x2,y2],...])    
    points = np.stack([pointX, pointY])
    points = points.T

    #end = time.time()
    #print("time:{}s".format(end-start))

    return points

# 凸包の関数により輪郭点抽出
def MakeContour(points):
    # 凸包
    # 入力は[[[1,2]], [[3,4]], ...]のような形で、floatなら32にする
    # (入力[[1,2],[3,4], ...]でもできた)
    points = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points)

    # 出力は[[[1,2]], [[3,4]], ...]のような形になるので
    # [[1,2],[3,4], ...]の形にreshape
    hull = np.reshape(hull, (hull.shape[0], 2))

    # 面積計算
    area = cv2.contourArea(hull)

    #print("hull:{}, area:{}".format(hull.shape[0], area))

    return hull, area

def PlotContour(hull, color="red"):
    # 点プロット
    X, Y = Disassemble2d(hull)
    plt.plot(X, Y, marker=".",linestyle="None",color=color)

    # hullを[0,1,2,..n] -> [1,2,...,n,0]の順番にしたhull2作成
    hull2 = list(hull[:])
    a = hull2.pop(0)
    hull2.append(a)
    hull2 = np.array(hull2)

    # hull2を利用して線を引く
    for a, b in zip(hull, hull2):
        LX, LY = line2d(a, b)
        plt.plot(LX, LY, color=color)

def buildAABB(points):
    #なんとこれで終わり
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    return max_p, min_p

#陰関数のグラフ描画
#fn  ...fn(x, y) = 0の左辺
#AABB_size ...AABBの各辺をAABB_size倍する
def plot_implicit2d(fn, points=None, AABB_size=1, bbox=(2.5,2.5), contourNum=30):

    # pointsの入力があればAABB生成してそれをもとにスケール設定
    if points is not None:
        #AABB生成
        max_p, min_p = buildAABB(points)

        xmax, ymax = max_p
        xmin, ymin = min_p

        #AABBの各辺がAABB_size倍されるように頂点を変更
        xmax = xmax + (xmax - xmin)/2 * AABB_size
        xmin = xmin - (xmax - xmin)/2 * AABB_size
        ymax = ymax + (ymax - ymin)/2 * AABB_size
        ymin = ymin - (ymax - ymin)/2 * AABB_size

    # pointsの入力がなければ適当なbboxでスケール設定
    else:
        xmin, xmax, ymin, ymax = bbox*2

    # スケールをもとにcontourNum刻みでX, Y作成
    X = np.linspace(xmin, xmax, contourNum)
    Y = np.linspace(ymin, ymax, contourNum)
    X, Y = np.meshgrid(X, Y)

    # fn(X, Y) = 0の等高線を引く
    #Z = fn(X,Y)
    Z = np.array([fn(X[i], Y[i]) for i in range(contourNum)])
    plt.contour(X, Y, Z, [0])

