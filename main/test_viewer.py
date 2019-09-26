import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from loadOBJ import loadOBJ
#from normal import normalEstimate
from OBB import buildOBB
from figure import sphere, plane

#seabornはimportしておくだけでもmatplotlibのグラフがきれいになる
import seaborn as sns
sns.set_style("darkgrid")

####最初にグラフの初期化？をする#############################

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#############################################################

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
def OBBViewer(vertices):
    #OBB作成
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

#球描画
def SphereViewer(p):
    #媒介変数
    theta = np.arange(-np.pi, np.pi, 0.01)
    phi = np.arange(-2*np.pi, 2*np.pi, 0.02)

    #二次元メッシュにしないといけない
    T, P = np.meshgrid(theta, phi)

    #パラメータ
    r0=[p[0], p[1], p[2]]       #中心座標
    r=p[3]                      #半径

    #球の方程式
    x = r * np.cos(T) * np.cos(P) + r0[0]
    y = r * np.cos(T) * np.sin(P) + r0[1]
    z = r * np.sin(T) + r0[2]

    #描画
    ax.scatter(r0[0] , r0[1] , r0[2],  color='yellow')
    ax.plot_wireframe(x, y, z,  linestyle='dotted', linewidth=0.3)

#平面描画
def PlaneViewer(p):
    print(p)
    

def OptiViewer(path, result, fig_type):

    #点群をobjファイルから読み込み
    vertices, X, Y, Z = loadOBJ(path)

    #点群を描画
    ax.plot(X,Y,Z,marker=".",linestyle='None',color="green")

    #OBBを描画
    OBBViewer(vertices)

    #最適化された図形を描画
    if fig_type == 0:
        SphereViewer(result)

    elif fig_type == 1:
        PlaneViewer(result)


    #最後に.show()を書いてグラフ表示
    plt.show()

OptiViewer("../data/pumpkin.obj", [-3.10735045,    1.81359686, -110.75950196, 37.29042182], 0)