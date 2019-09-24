import open3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from loadOBJ import loadOBJ
from OBB import buildOBB

#open3dから直接plyファイル読み込み
#pointcloud = open3d.read_point_cloud("airplane.ply")

#objファイルをnumpyで読み込んで,open3dのデータ形式に
pc, X, Y, Z = loadOBJ("./data/teapot.obj")
pointcloud = open3d.PointCloud()
pointcloud.points = open3d.Vector3dVector(pc)

# 法線推定
open3d.estimate_normals(
    pointcloud,
    search_param = open3d.KDTreeSearchParamHybrid(
        radius = 5, max_nn = 30))

# 法線の方向を視点ベースでそろえる
open3d.orient_normals_towards_camera_location(
    pointcloud,
    camera_location = np.array([0., 10., 10.], dtype="float64"))

#nキーで法線表示
#open3d.draw_geometries([pointcloud])

#法線,点群をnumpyへ変換
normal = np.asarray(pointcloud.normals)
#pc = np.asarray(pointcloud.points)


#xyzに分解
N = normal.T[:]
U = N[0, :]
V = N[1, :]
W = N[2, :]

"""
#グラフの枠を作っていく
fig = plt.figure()
ax = Axes3D(fig)

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#点群と法線をプロット
ax.plot(X,Y,Z,marker=",",linestyle='None',color="red")
#ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True)

#最後に.show()を書いてグラフ表示
plt.show()
"""

max_p, min_p, _ = buildOBB(pc) 

def sendData():
    return pc, normal, max_p, min_p