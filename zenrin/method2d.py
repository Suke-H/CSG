import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

class sphere:
    def __init__(self, p):
        # パラメータ
        # p = [x0, y0, r]
        self.p = p

    # 円の方程式: f(x,y) = r - √(x-a)^2 + (y-b)^2
    def f_rep(self, x, y):
        return self.p[2] - np.sqrt((x-self.p[0])**2 + (y-self.p[1])**2)

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
    #print(index)

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
    #print(index)

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



"""
C1 = sphere([0,0,1])

points1= ContourPoints(C1.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X1, Y1 = Disassemble2d(points1)
print("points1:{}".format(len(X1)))

points2= InteriorPoints(C1.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
X2, Y2 = Disassemble2d(points2)
print("points1:{}".format(len(X2)))

plt.plot(X1, Y1, marker="o",linestyle="None",color="blue")
plt.plot(X2, Y2, marker=".",linestyle="None",color="red")
MakeContour(points2)
plt.show()
"""