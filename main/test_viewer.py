import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from loadOBJ import loadOBJ
from OBB import buildOBB, buildAABB
import figure as F
from optimize import figOptimize

#seabornはimportしておくだけでもmatplotlibのグラフがきれいになる
import seaborn as sns
sns.set_style("darkgrid")

#点群データなどをx, y, zに分解する

#[x1, y1, z1]         [x1, x2, ..., xn]
#      :        ->    [y1, y2, ..., yn]
#[xn, yn, zn]         [z1, z2, ..., zn]
def Disassemble(XYZ):
    XYZ = XYZ.T[:]
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]

    return X, Y, Z


def line(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    z = a[2]*t + b[2]*(1-t)

    return x, y, z

#点群を入力としてOBBを描画する
def OBBViewer(ax, vertices):
    #OBB生成
    max_p, min_p, _ = buildOBB(vertices)

    #直積：[smax, smin]*[tmax, tmin]*[umax, umin] <=> 頂点
    s_axis = np.vstack((max_p[0], min_p[0]))
    t_axis = np.vstack((max_p[1], min_p[1]))
    u_axis = np.vstack((max_p[2], min_p[2]))

    products = np.asarray(list(itertools.product(s_axis, t_axis, u_axis)))
    vertices = np.sum(products, axis=1)

    #各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))


    #頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y, z = line(vertices[i], vertices[j])
                ax.plot(x,y,z,marker=".",color="orange")

    #OBBの頂点の1つ
    vert_max = min_p[0] + min_p[1] + max_p[2]
    vert_min = max_p[0] + max_p[1] + min_p[2]

    #xyzに分解
    Xmax, Ymax, Zmax = Disassemble(max_p)
    Xmin, Ymin, Zmin = Disassemble(min_p)

    #頂点なども描画
    ax.plot(Xmax,Ymax,Zmax,marker="X",linestyle="None",color="red")
    ax.plot(Xmin,Ymin,Zmin,marker="X",linestyle="None",color="blue")
    ax.plot([vert_max[0], vert_min[0]],[vert_max[1], vert_min[1]],[vert_max[2], vert_min[2]],marker="o",linestyle="None",color="black")

#陰関数のグラフ描画
#fn  ...fn(x, y, z) = 0の左辺
def plot_implicit(ax, fn, vertices):
    #AABB生成
    max_p, min_p = buildAABB(vertices)
    print(max_p, min_p)
    max_p = max_p*4
    min_p = min_p*4

    xmax, ymax, zmax = max_p[0], max_p[1], max_p[2]
    xmin, ymin, zmin = min_p[0], min_p[1], min_p[2]

    A = np.linspace(xmin, xmax, 100) #等高線の刻み
    B = np.linspace(xmin, xmax, 15) #等高線の数
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: #XY平面に等高線をプロット
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: #XZ平面に等高線をプロット
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: #YZ平面に等高線をプロット
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    #AABBの範囲に制限
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

def sh(x, y, z):
        return 37.29042182 - np.sqrt((x+3.10735045)**2 + (y-1.81359686)**2 + (z+110.75950196)**2)

#mainの部分
def OptiViewer(path, fig_type):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #点群をobjファイルから読み込み
    vertices, X, Y, Z = loadOBJ(path)

    #点群を描画
    ax.plot(X,Y,Z,marker=".",linestyle='None',color="green")

    #OBBを描画
    OBBViewer(ax, vertices)

    ###最適化###
    result = figOptimize(path, fig_type)
    print(result.x)

    #fig_typeに応じた図形を選択
    if fig_type==0:
        figure = F.sphere(result.x)
    elif fig_type==1:
        figure = F.plane(result.x)

    #最適化された図形を描画figure.f_rep_draw
    plot_implicit(ax, sh, vertices)

    #最後に.show()を書いてグラフ表示
    plt.show()

OptiViewer("../data/pumpkin.obj", 0)