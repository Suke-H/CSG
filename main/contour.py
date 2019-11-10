import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import numpy as np

from method import *
from PreProcess2 import PreProcess2
from figure_sample import *
from FigureDetection import CountPoints


def line2d(a, b):
    t = np.arange(0, 1, 0.01)
    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    return x, y

def MakeContour(points, plane):
    N = points.shape[0]
    # 平面のパラメータ
    a, b, c, d = plane.p
    n = np.array([a, b, c])

    # f(p0 + t n) = 0 <=> t = f(p0)/(a^2+b^2+c^2)
    X, Y, Z = Disassemble(points)
    t = plane.f_rep(X,Y,Z) / (a**2+b**2+c**2)
    #print(t.shape)
    tn = np.array([t[i]*n for i in range(N)])
    #print(tn.shape)
    # p = p0 + t n
    UVpoints = points + tn
    print(UVpoints.shape)

    # 新しい原点を適当に選んだ1点にする
    O = UVpoints[0]
    # 適当な2点のベクトルを軸の1つにする
    u = norm(UVpoints[1] - O)
    # v = u × n
    v = norm(np.cross(u, n))
    # UV座標に変換
    UVvector = np.array([[[np.dot((UVpoints[i]-O), u), np.dot((UVpoints[i]-O), v)]] for i in range(N)], dtype=np.float32)
    #print(UVpoints, UVpoints.shape)

    # 凸包
    # 入力は[[[1,2]], [[3,4]], ...]のような形で、floatなら32にする
    hull = cv2.convexHull(UVvector)

    # reshape
    UVvector = np.reshape(UVvector, (N, 2))
    hull = np.reshape(hull, (hull.shape[0], 2))
    print(hull.shape)

    # xyz座標に変換
    #XYZvector = [O + UVvector[i][0]*u + UVvector[i][1]*v for i in range(N)]
    XYZhull = np.array([O + hull[i][0]*u + hull[i][1]*v for i in range(hull.shape[0])])
    print(XYZhull.shape)

    return Disassemble(UVpoints), Disassemble(XYZhull), XYZhull


#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")


points, X, Y, Z, normals, length = PreProcess2()
figure = P_Z0
MX, MY, MZ, num, index = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi)
(UX, UY, UZ), (HX, HY, HZ), hull = MakeContour(points, P_Z0)


# [0,1,2,..n] -> [1,2,...,n,0]
hull2 = list(hull[:])
a = hull2.pop(0)
hull2.append(a)
hull2 = np.array(hull2)

#点群を描画
ax.plot(X,Y,Z,marker="o",linestyle='None',color="white")
#ax.plot(MX,MY,MZ,marker=".",linestyle='None',color="green")
ax.plot(UX,UY,UZ, marker=".",linestyle='None',color="blue")
ax.plot(HX,HY,HZ, marker="o",linestyle='None',color="red")
for a, b in zip(hull, hull2):
    LX, LY, LZ = line(a, b)
    ax.plot(LX, LY, LZ, color="red")


#軸
#ax.quiver([TX[0]], [TY[0]], [TZ[0]], [u[0]], [u[1]], [u[2]],  length=1,color='red', normalize=True)
#ax.quiver([TX[0]], [TY[0]], [TZ[0]], [v[0]], [v[1]], [v[2]],  length=1,color='blue', normalize=True)
#ax.quiver([TX[0]], [TY[0]], [TZ[0]], [n[0]], [n[1]], [n[2]],  length=1,color='green', normalize=True)

#図形
plot_implicit(ax, P_Z0.f_rep)

plt.show()
"""
UV = uv.T[:]
UX = UV[0, :]
UY = UV[1, :]
HULL = hull.T[:]
HX = HULL[0, :]
HY = HULL[1, :]


# [0,1,2,..n] -> [1,2,...,n,0]
hull2 = list(hull[:])
a = hull2.pop(0)
hull2.append(a)
hull2 = np.array(hull2)

plt.plot(UX,UY, marker="o",linestyle='None',color="green")
plt.plot(HX,HY, marker=".",linestyle='None',color="red")
for a, b in zip(hull, hull2):
    LX, LY = line2d(a, b)
    plt.plot(LX, LY, color="red")

plt.show()
"""