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

def line(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    z = a[2]*t + b[2]*(1-t)

    """
    p = a*t + b*(1-t)
    #xyzに分解
    N = p.T[:]
    x = N[0, :]
    y = N[1, :]
    z = N[2, :]
    """

    return x, y, z

#点群をobjファイルから読み込み
vertices, X, Y, Z = loadOBJ("../data/pumpkin.obj")

#法線推定
#normal = normalEstimate(vertices, 9)

#xyzに分解
#N = normal.T[:]
#U = N[0, :]
#V = N[1, :]
#W = N[2, :]

max_p, min_p, P = buildOBB(vertices) 
vert_max = min_p[0] + min_p[1] + max_p[2]
vert_min = max_p[0] + max_p[1] + min_p[2]
print(vert_max, vert_min)
#xyzに分解
MAX = max_p.T[:]
Xmax = MAX[0, :]
Ymax = MAX[1, :]
Zmax = MAX[2, :]
MIN = min_p.T[:]
Xmin = MIN[0, :]
Ymin = MIN[1, :]
Zmin = MIN[2, :]
"""
VMAX = vert_max.T[:]
VXmax = VMAX[0, :]
VYmax = VMAX[1, :]
VZmax = VMAX[2, :]
VMIN = vert_min.T[:]
VXmin = VMIN[0, :]
VYmin = VMIN[1, :]
VZmin = VMIN[2, :]
"""

#軸描画
sx, sy, sz = line(max_p[0], min_p[0])
tx, ty, tz = line(max_p[1], min_p[1])
ux, uy, uz = line(max_p[2], min_p[2])

#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#linestyle='None'にしないと初期値では線が引かれるが、3次元の散布図だと大抵ジャマになる
#markerは無難に丸
ax.plot(X,Y,Z,marker=".",linestyle='None',color="green")
ax.plot(Xmax,Ymax,Zmax,marker="X",linestyle="None",color="red")
ax.plot(Xmin,Ymin,Zmin,marker="X",linestyle="None",color="blue")
ax.plot([vert_max[0], vert_min[0]],[vert_max[1], vert_min[1]],[vert_max[2], vert_min[2]],marker="o",linestyle="None",color="black")


#ax.plot(sx,sy,sz,marker="o",color="orange")
#ax.plot(tx,ty,tz,marker="o",color="orange")
#ax.plot(ux,uy,uz,marker="o",color="orange")

#ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True)

########OBB描画##############################################################
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


####球描画テスト####

theta = np.arange(-np.pi, np.pi, 0.01)
phi = np.arange(-2*np.pi, 2*np.pi, 0.02)

#二次元メッシュにしないといけない
T, P = np.meshgrid(theta, phi)
#r0=[0.667871,0.0084,0.224349]
#r=3.014048083
#r0=[-7.17424154e-02,  1.13297596e+00, -1.00812392e-05]
#r=1.88807676e+00
#r0=[1.30615536e-01,  1.17631883e+00, -5.30004035e-04]
#r=2.03825280e+00
#r0=[2.30258079e-03, 1.38067327e+00, 8.50101505e-04]
#r=1.98709153e+00

#house
#r0=[130.04026077,   3.83085487, 135.60081717]
#r=274.23665354

#cow
#r0=[-5.05159977e-01,  1.19212621e-01, -2.75432093e-03]
#r=4.13813401e+00

#pumpkin
r0=[-3.10735045,    1.81359686, -110.75950196]
r=37.29042182
X, Y, Z = sphere(T, P, r, r0)

ax.scatter(r0[0] , r0[1] , r0[2],  color='yellow')
ax.plot_wireframe(X, Y, Z, linestyle='dotted', linewidth=0.3)

###平面描画テスト###
x = np.arange(-3, 3, 0.01)
y = np.arange(0, 3, 0.01)
X, Y = np.meshgrid(x, y)
#teapot
#n=[-0.00223537,  0.0012511 ,  0.99999672]
#d=0.00180294

#pumpkin
n=[0.1611404 ,  0.98440921, -0.07051438]
d=9.21609631


#house
#n=[6.85153886e-01, 7.28392574e-01, 2.90024015e-03]
#d=2.19640328e+01
Z = plane(X, Y, n, d)
#ax.plot_surface(X,Y,Z,alpha=0.3)

#np.savetxt('sample_2.txt', Z)


#最後に.show()を書いてグラフ表示
plt.show()