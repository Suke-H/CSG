import numpy as np

from method2d import *
import figure2d as F
from ClossNum import CheckClossNum, CheckClossNum2


def CheckIB(child, fig, max_p, min_p, l):
    # 円
    if fig==0:
        x, y, r = child
        w, h = l/2, l/2
    # 正三角形
    elif fig==1:
        x, y, r, _= child
        w, h = l/2, l/2
    # 長方形
    elif fig==2:
        x, y, w, h, _ = child
        r = l/2

    if (min_p[0] < x < max_p[0]) and (min_p[1] < y < max_p[1]) and (l*0.2 < r < l) and (l*0.2 < w < l) and (l*0.2 < h < l):
        return True

    else:
        return False

def CheckIB2(child, fig, max_p, min_p, l):
    # 円
    if fig==0:
        x, y, r = child
        w, h = l/2, l/2
    # 正三角形
    elif fig==1:
        x, y, r, _= child
        w, h = l/2, l/2
    # 長方形
    elif fig==2:
        x, y, w, h, _ = child
        r = l/2

    if (min_p[0] < x < max_p[0]) and (min_p[1] < y < max_p[1]) and (l/15< r < l) and (l/15 < w < l) and (l/15 < h < l):
        return True

    else:
        return False


# outの枠内に図形が入っているかのチェック
# contoursはoutの輪郭点の点群配列
# def CheckIB2(figure, contour, max_p, min_p):
#     # 図形の輪郭点を取る
#     AABB = [min_p[0], max_p[0], min_p[1], max_p[1]]
#     points = ContourPoints(figure.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01)
#     #print(points.shape)
#
#     # X1, Y1 = Disassemble2d(points)
#     # X2, Y2 = Disassemble2d(contour)
#     # plt.plot(X2, Y2, marker=".",linestyle="None",color="red")
#     # plt.plot(X1, Y1, marker="o",linestyle="None",color="orange")
#     # plt.show()
#
#     # 図形の輪郭点がoutの枠内に入ってるかチェック
#     inside = np.array([CheckClossNum(points[i], contour) for i in range(points.shape[0])])
#
#     if np.all(inside):
#         return True
#
#     else:
#         return False

# なし
def CalcIoU0(points, out_contour, out_area, figure, flag=False):

    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    # if not CheckIB(figure.p, fig, max_p, min_p, l):
    #     return -100

    # outの枠内にあるかチェック
    #if not CheckIB2(figure, out_contour, max_p, min_p):
        #return -100

    ########################################################

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W>=0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    
    Cout = points.shape[0] - Cin

    # Aout = out_shapeの面積 - Ain
    Aout = out_area - Ain

    if Aout < 0:
        return -100

    if flag==True:
        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin/Ain - Cout/Aout
    #return Cin/(Ain+np.sqrt(Cin)) - Cout/(Aout+np.sqrt(Cout))

# checkIB
def CalcIoU1(points, out_contour, out_area, figure, flag=False):

    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    if not CheckIB(figure.p, fig, max_p, min_p, l):
        return -100

    # outの枠内にあるかチェック
    #if not CheckIB2(figure, out_contour, max_p, min_p):
        #return -100

    ########################################################

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W>=0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    
    Cout = points.shape[0] - Cin

    # Aout = out_shapeの面積 - Ain
    Aout = out_area - Ain

    if Aout < 0:
        return -100

    if flag==True:
        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin/Ain - Cout/Aout
    #return Cin/(Ain+np.sqrt(Cin)) - Cout/(Aout+np.sqrt(Cout))

# Score2
def CalcIoU2(points, out_contour, out_area, figure, flag=False):

    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    # if not CheckIB(figure.p, fig, max_p, min_p, l):
    #     return -100

    # outの枠内にあるかチェック
    #if not CheckIB2(figure, out_contour, max_p, min_p):
        #return -100

    ########################################################

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W>=0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    
    Cout = points.shape[0] - Cin

    # Aout = out_shapeの面積 - Ain
    Aout = out_area - Ain

    if Aout < 0:
        return -100

    if flag==True:
        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    #return Cin/Ain - Cout/Aout
    return Cin/(Ain+np.sqrt(Cin)) - Cout/(Aout+np.sqrt(Cout))

# 1 + 2
def CalcIoU3(points, out_contour, out_area, figure, flag=False):

    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    if not CheckIB(figure.p, fig, max_p, min_p, l):
        return -100

    # outの枠内にあるかチェック
    #if not CheckIB2(figure, out_contour, max_p, min_p):
        #return -100

    ########################################################

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W>=0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    
    Cout = points.shape[0] - Cin

    # Aout = out_shapeの面積 - Ain
    Aout = out_area - Ain

    if Aout < 0:
        return -100

    if flag==True:
        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    #return Cin/Ain - Cout/Aout
    return Cin/(Ain+np.sqrt(Cin)) - Cout/(Aout+np.sqrt(Cout))

def calc_score2(points, out_contour, out_area, figure, flag=False):
    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    # print(max_p, min_p)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    if not CheckIB2(figure.p, fig, max_p, min_p, l):
        return -100

    ########################################################

    # figureの2倍の図形を作成し、それを外枠とする
    if fig == 0:
        out_p = figure.p[:]
        out_p[2] *= 2
        out_figure = F.circle(out_p)

    elif fig == 1:
        out_p = figure.p[:]
        out_p[2] *= 2
        out_figure = F.tri(out_p)

    else:
        out_p = figure.p[:]
        out_p[2] *= 2
        out_p[3] *= 2
        out_figure = F.rect(out_p)

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W >= 0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    X2 = np.delete(X, in_index)
    Y2 = np.delete(Y, in_index)
    W2 = out_figure.f_rep(X2, Y2)
    out_index = np.where(W2 >= 0)
    Cout = len(out_index[0])

    # Aout = outの面積 - Ain
    Aout = out_figure.CalcArea() - Ain

    if flag == True:
        X = X[in_index]
        Y = Y[in_index]
        plt.plot(X, Y, marker=".", linestyle="None", color="blue")
        X2 = X2[out_index]
        Y2 = Y2[out_index]
        plt.plot(X2, Y2, marker=".", linestyle="None", color="red")

        fig1 = ContourPoints(figure.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01)
        Xin, Yin = Disassemble2d(fig1)
        plt.plot(Xin, Yin, marker=".", linestyle="None", color="blue")
        fig2 = ContourPoints(out_figure.f_rep, AABB=AABB,grid_step=1000, epsilon=0.01)
        Xout, Yout = Disassemble2d(fig2)
        plt.plot(Xout, Yout, marker=".", linestyle="None", color="red")

        plt.show()

        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin / Ain - Cout / Aout

def calc_score1_5(points, out_contour, out_area, figure, flag=False):
    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    # print(max_p, min_p)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    if not CheckIB2(figure.p, fig, max_p, min_p, l):
        return -100

    ########################################################

    # figureの2倍の図形を作成し、それを外枠とする
    if fig == 0:
        out_p = figure.p[:]
        out_p[2] *= 1.5
        out_figure = F.circle(out_p)

    elif fig == 1:
        out_p = figure.p[:]
        out_p[2] *= 1.5
        out_figure = F.tri(out_p)

    else:
        out_p = figure.p[:]
        out_p[2] *= 1.5
        out_p[3] *= 1.5
        out_figure = F.rect(out_p)

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W >= 0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    X2 = np.delete(X, in_index)
    Y2 = np.delete(Y, in_index)
    W2 = out_figure.f_rep(X2, Y2)
    out_index = np.where(W2 >= 0)
    Cout = len(out_index[0])

    # Aout = outの面積 - Ain
    Aout = out_figure.CalcArea() - Ain

    if flag == True:
        X = X[in_index]
        Y = Y[in_index]
        plt.plot(X, Y, marker=".", linestyle="None", color="blue")
        X2 = X2[out_index]
        Y2 = Y2[out_index]
        plt.plot(X2, Y2, marker=".", linestyle="None", color="red")

        fig1 = ContourPoints(figure.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01)
        Xin, Yin = Disassemble2d(fig1)
        plt.plot(Xin, Yin, marker=".", linestyle="None", color="blue")
        fig2 = ContourPoints(out_figure.f_rep, AABB=AABB,grid_step=1000, epsilon=0.01)
        Xout, Yout = Disassemble2d(fig2)
        plt.plot(Xout, Yout, marker=".", linestyle="None", color="red")

        plt.show()

        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin / Ain - Cout / Aout

def calc_score1_2(points, out_contour, out_area, figure, flag=False):
    # AABB内にあるのかチェック
    max_p, min_p, _, l, AABB_area = buildAABB2d(points)

    # print(max_p, min_p)

    if len(figure.p) == 3:
        fig = 0
    elif len(figure.p) == 4:
        fig = 1
    else:
        fig = 2

    if not CheckIB2(figure.p, fig, max_p, min_p, l):
        return -100

    ########################################################

    # figureの2倍の図形を作成し、それを外枠とする
    if fig == 0:
        out_p = figure.p[:]
        out_p[2] *= 1.2
        out_figure = F.circle(out_p)

    elif fig == 1:
        out_p = figure.p[:]
        out_p[2] *= 1.2
        out_figure = F.tri(out_p)

    else:
        out_p = figure.p[:]
        out_p[2] *= 1.2
        out_p[3] *= 1.2
        out_figure = F.rect(out_p)

    # W >= 0(-図形の内部)のインデックスを取り出し、
    # その数をCinとして保存
    X, Y = Disassemble2d(points)
    W = figure.f_rep(X, Y)
    in_index = np.where(W >= 0)
    Cin = len(in_index[0])

    # Ain = 推定図形の面積
    Ain = figure.CalcArea()

    # Cout = outの点群数
    # (外枠内の点群数 - inの点群数)
    X2 = np.delete(X, in_index)
    Y2 = np.delete(Y, in_index)
    W2 = out_figure.f_rep(X2, Y2)
    out_index = np.where(W2 >= 0)
    Cout = len(out_index[0])

    # Aout = outの面積 - Ain
    Aout = out_figure.CalcArea() - Ain

    if flag == True:
        X = X[in_index]
        Y = Y[in_index]
        plt.plot(X, Y, marker=".", linestyle="None", color="blue")
        X2 = X2[out_index]
        Y2 = Y2[out_index]
        plt.plot(X2, Y2, marker=".", linestyle="None", color="red")

        fig1 = ContourPoints(figure.f_rep, AABB=AABB, grid_step=1000, epsilon=0.01)
        Xin, Yin = Disassemble2d(fig1)
        plt.plot(Xin, Yin, marker=".", linestyle="None", color="blue")
        fig2 = ContourPoints(out_figure.f_rep, AABB=AABB,grid_step=1000, epsilon=0.01)
        Xout, Yout = Disassemble2d(fig2)
        plt.plot(Xout, Yout, marker=".", linestyle="None", color="red")

        plt.show()

        print(points.shape)
        print("{}/{} - {}/{}".format(Cin, Ain, Cout, Aout))

    return Cin / Ain - Cout / Aout

# 目標図形と最適図形でIoUを算出
def LastIoU(goal, opti, AABB, path):
    # Figureの初期化
    #fig = plt.figure(figsize=(12, 8))
    
    # intersectionの面積
    inter_fig = F.inter(goal, opti)
    inter_points = ContourPoints(inter_fig.f_rep, AABB=AABB, grid_step=1000)
    inter_points = np.array(inter_points, dtype=np.float32)

    inter_area = cv2.contourArea(cv2.convexHull(inter_points))

    # unionの面積
    union_fig = F.union(goal, opti)
    union_points = ContourPoints(union_fig.f_rep, AABB=AABB, grid_step=1000)
    union_points = np.array(union_points, dtype=np.float32)
    union_area = cv2.contourArea(cv2.convexHull(union_points))

    X1, Y1 = Disassemble2d(inter_points)
    X2, Y2 = Disassemble2d(union_points)

    plt.plot(X1, Y1, marker=".",linestyle="None",color="red")
    plt.plot(X2, Y2, marker=".",linestyle="None",color="yellow")
    plt.savefig(path)
    plt.close()

    if len(goal.p) != len(opti.p):
        return -1

    if inter_points.shape[0] <= 2:
        return -2

    # print("{} / {}".format(inter_area, union_area))

    return inter_area / union_area
