import numpy as np
import matplotlib.pyplot as plt

from method2d import *

#pointsの各点が輪郭点contour内にあるかの判定
def CheckClossNum(p, contour):
    # 輪郭点の辺をつくるため、
    # 輪郭点を[0,1,2,..n] -> [1,2,...,n,0]の順にした配列を作成
    order = [i for i in range(1, contour.shape[0])]
    order.append(0)
    contour2 = contour[order, :]

    # l: pから伸ばした左にx軸平行な半直線
    # 各辺とlの交差数をカウントする
    #judge_list = []
    crossCount = 0
    #for p, a, b in zip(points, contour, contour2):
    for a, b in zip(contour, contour2):
        # a,bのy座標がどちらもpのy座標より小さい, 大きい, a,bのx座標がどちらもpのx座標より大きい => 交差しない
        if (a[1]<p[1] and b[1]<p[1]) or (a[1]>p[1] and b[1]>p[1]) or (a[0]>p[0] and b[0]>p[0]):
            continue
        # a,bのx座標がどちらもpのx座標より小さい =>　交差する
        if a[0]<p[0] and b[0]<p[0]:
            crossCount+=1
            continue
        # lが直線として,辺との交点cのx座標を求める
        cx = (p[1]*(a[0]-b[0]) + a[1]*b[0] - a[0]*b[1]) / (a[1]-b[1])
        # cx < pxなら交差する
        if cx < p[0]:
            crossCount+=1

    # 交差数が偶数なら外、奇数なら内
    if crossCount%2 == 0:
        return False
    else:
        return True

# n = 2000
# x = (np.random.rand(n) - 0.5)*2.5
# y = (np.random.rand(n) - 0.5)*2.5
# x1, y1 = [], []
    
# # define a polygon
# for i in range(5):
#     x1.append(np.cos(i*2.*np.pi/5.))
#     y1.append(np.sin(i*2.*np.pi/5.))

# points = Composition2d(x, y)
# contour = Composition2d(x1, y1)

# inside = np.array([CheckClossNum(points[i], contour) for i in range(points.shape[0])])

# x1.extend([x1[0]])
# y1.extend([y1[0]])

# plt.plot(x[inside], y[inside], marker=".",linestyle="None",color="red")
# plt.plot(x[inside==False], y[inside==False], marker=".",linestyle="None",color="black")
# plt.plot(x1, y1)
# plt.savefig("data/inpoly.png")
# plt.show()