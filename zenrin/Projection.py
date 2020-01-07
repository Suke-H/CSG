import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import numpy as np

# zenrin
from method2d import *

# main
import figure2 as F
from method import *

def PlaneProjection(points, plane):
    N = points.shape[0]

    # 平面のパラメータ
    a, b, c, d = plane.p
    n = np.array([a, b, c])

    # 法線方向に点を動かして平面に落とし込む
    # f(p0 + t n) = 0 <=> t = f(p0)/(a^2+b^2+c^2)
    X, Y, Z = Disassemble(points)
    t = plane.f_rep(X,Y,Z) / (a**2+b**2+c**2)

    # p = p0 + t n
    tn = np.array([t[i]*n for i in range(N)])
    plane_points = points + tn
    print(plane_points.shape)

    # 新しい原点を適当に選んだ1点にする
    O = plane_points[0]
    # 適当な2点のベクトルを軸の1つにする
    u = norm(plane_points[1] - O)
    # v = u × n
    v = norm(np.cross(u, n))
    # UV座標に変換
    UVvector = np.array([[np.dot((plane_points[i]-O), u), np.dot((plane_points[i]-O), v)]for i in range(N)])
    print(UVvector.shape)

    # reshape
    #UVvector = np.reshape(UVvector, (N, 2))

    # 平面に落とした点の"3次元"座標、"2次元"座標
    return plane_points, UVvector