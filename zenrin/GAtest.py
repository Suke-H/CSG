import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import re
import time

# main
from method import *
from PreProcess import NormalEstimate
from ransac import RANSAC
from Projection import Plane2DProjection, Plane3DProjection

# zenrin
import figure2d as F
#from IoUtest import CalcIoU, CalcIoU2, LastIoU
from GA import *
from MakeDataset import MakePointSet, MakePointSet3D
from TransPix import MakeOuterFrame
from ClossNum import CheckClossNum, CheckClossNum2, CheckClossNum3

def PlaneDetect(points):

    #法線, 取得
    normals = NormalEstimate(points)

    _, _, l = buildOBB(points)
    print("l:{}".format(l))

    # 平面検出
    # index: pointsからフィットした点のインデックス
    plane, index, num = RANSAC(points, normals, epsilon=l*0.05, alpha=np.pi/8)

    selectIndex = np.array([True if i in index[0] else False for i in range(points.shape[0])])

    # フィット点を平面射影
    # plane_points: 射影後の3d座標点群
    # UVvector: 射影後の2d座標点群
    plane_points, UVvector, u, v, O = Plane2DProjection(points[index], plane)

    # プロット準備
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # 点群描画
    X, Y, Z = Disassemble(points)
    MX, MY, MZ = X[index], Y[index], Z[index]
    PX, PY, PZ = Disassemble(plane_points)
    ax.plot(X, Y, Z, marker="o", linestyle='None', color="white")
    ax.plot(MX, MY, MZ, marker=".", linestyle='None', color="red")
    ax.plot(PX, PY, PZ, marker=".", linestyle='None', color="blue")
    # 平面描画
    plot_implicit(ax, plane.f_rep, points, AABB_size=1, contourNum=15)

    plt.show()
    plt.close()

    # 射影2d点群描画
    UX, UY = Disassemble2d(UVvector)
    plt.plot(UX, UY, marker="o",linestyle="None",color="red")

    plt.show()
    plt.close()

    return UVvector, plane, u, v, O, selectIndex

# input: [1,1,1,1,0,1,1,0]
# output: [1,1,1,1,0,1]
#                    
# result: [1,1,1,1,0,0,1,0]
#          ^ ^ ^ ^   ^ ^
def SelectIndex(input, output):

    result = []
    j = 0
    for i in input:
        if i == 0:
            result.append(False)

        else:
            if output[j] == 1:
                result.append(True)
            else:
                result.append(False)

    return np.array(result)

# TP:1, TN:2, FP:3, FN:4

#    true  opti
# TP: 1  ->  1
# TF: 0  ->  0
# FP: 0  ->  1
# FN: 1  ->  0
def ConfusionLabeling(trueIndex, optiIndex):
    confusionIndex = []

    for (true, opti) in zip(trueIndex, optiIndex):
        if true==1 and opti==1:
            confusionIndex.append(1)
        elif true==0 and opti==0:
            confusionIndex.append(2)
        elif true==0 and opti==1:
            confusionIndex.append(3)
        elif true==1 and opti==0:
            confusionIndex.append(4)
        else:
            print("そんなことある？")

    return np.array(confusionIndex)

def write_dataset(fig_type, num, dir_path="data/dataset/tri/"):

    fig_list = []
    AABB_list = []
    out_area_list = []
    points_list = []
    out_points_list = []

    for i in range(num):
        rate = Random(0.5, 1)
        # 2d図形点群作成
        fig, sign2d, AABB = MakePointSet(fig_type, 500, rate=rate)

        # 外枠作成
        out_points, out_area = MakeOuterFrame(sign2d, fig.CalcArea(), path=dir_path+"contour/"+str(i)+".png", 
        dilate_size=30, close_size=0, open_size=50, add_size=50)

        # 外枠内の点群だけにする
        inside = np.array([CheckClossNum3(sign2d[i], out_points) for i in range(sign2d.shape[0])])
        #inside = CheckClossNum2(sign2d, out_points)
        sign2d = sign2d[inside]

        fig_list.append(fig.p)
        AABB_list.append(AABB)
        out_area_list.append(out_area)
        # points_list.append(sign2d)
        # out_points_list.append(out_points)
        np.save(dir_path+"points/"+str(i), np.array(sign2d))
        np.save(dir_path+"outPoints/"+str(i), np.array(out_points))

        print("p:{}".format(fig.p))
        print("AABB:{}".format(AABB))
        print("outArea:{}".format(out_area))

        print("points{}:{}".format(i, sign2d.shape))
        print("outPoints{}:{}".format(i, out_points.shape))

    print("fig:{}".format(np.array(fig_list).shape))
    print("AABB:{}".format(np.array(AABB_list).shape))
    print("outArea:{}".format(np.array(out_area_list).shape))

    np.save(dir_path+"fig", np.array(fig_list))
    np.save(dir_path+"AABB", np.array(AABB_list))
    np.save(dir_path+"outArea", np.array(out_area_list))

def use_dataset(fig_type, num, dir_path="data/dataset/tri/", out_path="data/GAtest/tri/"):
    # データセット読み込み
    fig_list = np.load(dir_path+"fig.npy")
    AABB_list = np.load(dir_path+"AABB.npy")
    outArea_list = np.load(dir_path+"outArea.npy")

    print("fig:{}".format(np.array(fig_list).shape))
    print("AABB:{}".format(np.array(AABB_list).shape))
    print("outArea:{}".format(np.array(outArea_list).shape))

    # points, outPointsはまずパスを読み込み
    points_paths = sorted(glob(dir_path + "points/**.npy"),\
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))
    outPoints_paths = sorted(glob(dir_path + "outPoints/**.npy"),\
                        key=lambda s: int(re.findall(r'\d+', s)[len(re.findall(r'\d+', s))-1]))

    # print(points_paths)
    # print(outPoints_paths)

    for i in range(num):
        print("epoch:{}".format(i))

        # points, outPoints読み込み
        points = np.load(points_paths[i])
        outPoints = np.load(outPoints_paths[i])

        # print("points{}:{}".format(i, points.shape))
        # print("outPoints{}:{}".format(i, outPoints.shape))

        # 他も参照
        fig_p = fig_list[i]
        AABB = AABB_list[i]
        outArea = outArea_list[i]

        # print("p:{}".format(fig_p))
        # print("AABB:{}".format(AABB))
        # print("outArea:{}".format(outArea))

        if fig_type==0:
            fig = F.circle(fig_p)
        elif fig_type==1:
            fig = F.tri(fig_p)
        else:
            fig = F.rect(fig_p)

        # GAにより最適パラメータ出力
        
        # step1
        # best0 = EntireGA(points, outPoints, outArea, CalcIoU0, out_path+"score0/"+str(i)+".png")
        # best1 = EntireGA(points, outPoints, outArea, CalcIoU1, out_path+"score1/"+str(i)+".png")
        # best2 = EntireGA(points, outPoints, outArea, CalcIoU2, out_path+"score2/"+str(i)+".png")
        # best3 = EntireGA(points, outPoints, outArea, CalcIoU3, out_path+"score3/"+str(i)+".png")

        # step1.5
        best0, n = EntireGA(points, outPoints, outArea, CalcIoU3, out_path+str(i)+".png")

        # step2
        # best0 = EntireGA(points, outPoints, outArea, CalcIoU3, out_path+"score0/"+str(i)+".png")
        # best1 = EntireGA(points, outPoints, outArea, CalcIoU3, out_path+"score1/"+str(i)+".png",
        #                 half_reset_num=15, all_reset_num=9)
        # best2 = EntireGA(points, outPoints, outArea, CalcIoU3, out_path+"score2/"+str(i)+".png",
        #                 add_num=30)
        # best3 = EntireGA(points, outPoints, outArea, CalcIoU3, out_path+"score3/"+str(i)+".png", 
        #                 add_num=30, half_reset_num=15, all_reset_num=9)

        IoU0 = LastIoU(fig, best0.figure, AABB)

        print([IoU0, n])

        with open(out_path+"IoU.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([IoU0, n])




# def test3D(fig_type, loop):
#     count = 0

#     while count != loop:
#         # 2d図形点群作成
#         para3d, sign3d, AABB3d, trueIndex = MakePointSet3D(fig_type, 500, rate=0.8)

#         # 平面検出, 2d変換
#         sign2d, plane, u, v, O, index1 = PlaneDetect(sign3d)

#         # 外枠作成
#         out_points, out_area = MakeOuterFrame(sign2d, path="data/GAtest/" + str(count) + ".png")

#         # 外枠内の点群だけにする
#         index2 = np.array([CheckClossNum3(sign2d[i], out_points) for i in range(sign2d.shape[0])])
#         #inside = CheckClossNum2(sign2d, out_points)
#         sign2d = sign2d[index2]

#         # GAにより最適パラメータ出力
#         #best = GA(sign)
#         best = EntireGA(sign2d, out_points, out_area)
#         print("="*50)

#         X, Y = Disassemble2d(sign2d)
#         index3 = (best.figure.f_rep(X, Y) >= 0)

#         estiIndex = SelectIndex(index1, SelectIndex(index2, index3))

#         print(best[fig_type].figure.p)

#         count+=1

#write_dataset(0, 50, dir_path="data/dataset/circle4/")
use_dataset(1, 50, dir_path="data/dataset/tri4/", out_path="data/GAtest/step1.5/")
#test2D(1, 3, "data/GAtest/IoU.csv")