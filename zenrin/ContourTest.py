import numpy as np
import re
from glob import glob
import csv

import figure2d as F
from method2d import *
from TransPix import MakeOuterFrame
from MakeDataset import MakePointSet

def Record(fig_type, dir_path):

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

    print(points_paths)
    print(outPoints_paths)

    num = len(fig_list)

    for i in range(num):

        # points, outPoints読み込み
        points = np.load(points_paths[i])
        outPoints = np.load(outPoints_paths[i])
        # 他も参照
        fig_p = fig_list[i]
        AABB = AABB_list[i]
        outArea = outArea_list[i]

        if fig_type==0:
            fig = F.circle(fig_p)
        elif fig_type==1:
            fig = F.tri(fig_p)
        else:
            fig = F.rect(fig_p)

        # AABBの面積、figの面積、pointsの数を記録
        umin, umax, vmin, vmax = AABB
        AABBArea = abs((umax-umin)*(vmax-vmin))
        figArea = fig.CalcArea()
        pointNum = points.shape[0]
        rate = figArea/AABBArea

        with open(dir_path+"circle4.csv", 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([AABBArea, figArea, pointNum, rate, pointNum*rate, pointNum/rate])

def test(points, fig, strings):
    # out_points, out_area = MakeOuterFrame(sign2d, path=dir_path+"contour/"+str(i)+".png")
    out_points, out_area = MakeOuterFrame(points, fig.CalcArea(), path="data/Contour/"+strings+".png"
    , dilate_size=30, close_size=0, open_size=50, add_size=10)

Record(0, "data/dataset/circle4/")
# fig_type = 0
# num = 500
# rate = 0.5
# fig, points, AABB = MakePointSet(fig_type, num, rate=rate)
# # points = np.load("data/dataset/circle2/points/29.npy")
# strings = str(fig_type) + "_" + str(num) + "_" + str(rate)
# test(points, fig, strings)