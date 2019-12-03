import numpy as np

from method2d import *

# 標識の点群pointsと図形figureとの一致度を計算する
def CalcIoU(points, figure):

    ### AND ######################################

    X, Y = Disassemble2d(points)

    # pointsを図形の関数に代入
    W = figure.f_rep(X, Y)

    # W > 0(=図形の内部)のインデックスを取り出し、
    # その数をand_numとして保存
    index = np.where(W>0)
    print(index)
    and_num = len(index[0])
    print("and_num:{}".format(and_num))

    # plot
    pointX = np.array([X[i] for i in index])
    pointY = np.array([Y[i] for i in index])

    plt.plot(pointX, pointY, marker="o",linestyle="None",color="blue")

    ### OR ######################################

    # 標識のhullを取る
    hull1, _ = MakeContour(points)

    # 図形から多めに輪郭点作成
    points2= ContourPoints(figure.f_rep, grid_step=200, epsilon=0.01, down_rate = 0.5)
    print("points2:{}".format(points2.shape[0]))

    # 標識 or 図形 の点群をまとめる
    or_points = np.concatenate([hull1, points2])
    print("or_points:{}".format(or_points.shape[0]))

    X2, Y2= Disassemble2d(or_points)
    plt.plot(X2, Y2, marker=".",linestyle="None",color="red")

    # 標識 or 図形 の面積計算
    hull, or_area = MakeContour(or_points)
    print("or_area:{}".format(or_area))

    PlotContour(hull, color="red")

    # IoU = AND(点群数) / OR(面積) とする
    return and_num / or_area

C1 = sphere([0,0,1])
C2 = sphere([0,1,1])
P1 = InteriorPoints(C1.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
print("points:{}".format(P1.shape[0]))

X1, Y1 = Disassemble2d(P1)
#points = ContourPoints(C2.f_rep, grid_step=300, epsilon=0.01, down_rate = 0.5)
#X2, Y2= Disassemble2d(points)

#plt.plot(X1, Y1, marker=".",linestyle="None",color="blue")
#plt.plot(X2, Y2, marker="o",linestyle="None",color="red")

print("IoU:{}".format(CalcIoU(P1, C2)))
plt.show()