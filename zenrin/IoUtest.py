import numpy as np

from method2d import *
import figure2d as F

# 標識の点群pointsと図形figureとの一致度を計算する
def CalcIoU(points, figure, flag=False):

    ### AND ######################################

    X, Y = Disassemble2d(points)

    # pointsを図形の関数に代入
    W = figure.f_rep(X, Y)

    # W > 0(=図形の内部)のインデックスを取り出し、
    # その数をand_numとして保存
    index = np.where(W>0)
    #print(index)
    and_num = len(index[0])
    #print("and_num:{}".format(and_num))


    ### OR ######################################

    # 標識のhullを取る
    hull1, _ = MakeContour(points)

    # 図形から多めに輪郭点作成
    points2= ContourPoints(figure.f_rep, bbox=(-8, 8), grid_step=1000, epsilon=0.01, down_rate = 0.5)
    #print("points2:{}".format(points2.shape[0]))

    # 標識 or 図形 の点群をまとめる
    or_points = np.concatenate([hull1, points2])
    #print("or_points:{}".format(or_points.shape[0]))

    # 標識 or 図形 の面積計算
    hull, or_area = MakeContour(or_points)
    #print("or_area:{}".format(or_area))

    # plot
    if flag == True:

        pointX = np.array([X[i] for i in index])
        pointY = np.array([Y[i] for i in index])

        plt.plot(pointX, pointY, marker="o",linestyle="None",color="blue")

        X2, Y2= Disassemble2d(or_points)
        plt.plot(X2, Y2, marker=".",linestyle="None",color="red")

        PlotContour(hull, color="red")

        plt.show()

    # IoU = AND(点群数) / OR(面積) とする
    return and_num / or_area

# Score = Cin/(Ain+√Cin) - Cout/Aout
def CalcIoU2(points, figure, flag=False):

    X, Y = Disassemble2d(points)

    # pointsを図形の関数に代入
    W = figure.f_rep(X, Y)

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    index = np.where(W>=0)
    Cin = len(index[0])

    # Ainに図形の面積
    Ain = figure.CalcArea()

    # Cout = 全点群数 - Cin
    Cout = points.shape[0] - Cin

    # AABB作成, 面積も計算
    max_p, min_p, _, _, _ = buildAABB(points)
    AABB_area = abs((max_p[0]-min_p[0]) * (max_p[1]-min_p[1]))* 9

    # Aout = AABBの面積 - Ain
    Aout = AABB_area - Ain

    # Ain = Ain * (√Cin / (AABB/a))
    #Ain = Ain*(np.sqrt(Cin))/(Aout/20)

    #print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))
    #x = np.pi*Ain/(2*Aout) + np.pi/4
    #y = Cin*np.cos(x)
    #print("{} -> {}".format(x, y))

    if flag==True:
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin/(Ain+np.sqrt(Cin)) - Cout/Ain

# 目標図形と最適図形でIoUを算出
def LastIoU(goal, opti, AABB):
    
    # intersectionの面積
    inter_fig = F.inter(goal, opti)
    inter_points = ContourPoints(inter_fig.f_rep, AABB=AABB)
    inter_points = np.array(inter_points, dtype=np.float32)
    inter_area = cv2.contourArea(cv2.convexHull(inter_points))

    # unionの面積
    union_fig = F.union(goal, opti)
    union_points = ContourPoints(union_fig.f_rep, AABB=AABB)
    union_points = np.array(union_points, dtype=np.float32)
    union_area = cv2.contourArea(cv2.convexHull(union_points))

    X1, Y1 = Disassemble2d(inter_points)
    X2, Y2 = Disassemble2d(union_points)

    plt.plot(X1, Y1, marker=".",linestyle="None",color="red")
    plt.plot(X2, Y2, marker=".",linestyle="None",color="yellow")
    plt.show()

    print("{} / {}".format(inter_area, union_area))

    return inter_area / union_area
