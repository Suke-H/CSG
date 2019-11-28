import numpy as np
from scipy.optimize import minimize
import open3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from loadOBJ import loadOBJ
from method import *
from figure_sample import *
import figure2 as F

#from test_viewer import plot_implicit

def PreProcess2():
	#自作した点群を読み込み
    points, X, Y, Z = MakePoints(CONE.f_rep, grid_step=100, epsilon=0.03, down_rate = 0.5)

    print("points:{}".format(len(X)))

    #点群をnp配列⇒open3d形式に
    pointcloud = open3d.PointCloud()
    pointcloud.points = open3d.Vector3dVector(points)

	# 法線推定
    open3d.estimate_normals(
    	pointcloud,
    	search_param = open3d.KDTreeSearchParamHybrid(
    	radius = 1, max_nn = 30))

	# 法線の方向を視点ベースでそろえる
    open3d.orient_normals_towards_camera_location(
        pointcloud,
        camera_location = np.array([0., 10., 10.], 
        dtype="float64"))

	#nキーで法線表示
    #open3d.draw_geometries([pointcloud])

	#法線をnumpyへ変換
    normals = np.asarray(pointcloud.normals)
	#OBB生成
	#(最適化の条件にも使いたい)
    _, _, length = buildOBB(points)
    print("OBB_length: {}".format(length))
    
    return points, X, Y, Z, normals, length

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#軸にラベルを付けたいときは書く
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

#下準備
points, X, Y, Z, normals, length = PreProcess2()
ax.plot(X, Y, Z, marker="o",linestyle="None",color="blue")
#figure = F.plane([0,0,1,0])
#figure = F.sphere([0.75, 0.75, 0.75, 0.75])

#count, X, Y, Z, _ = CountPoints(figure, X, Y, Z, normals)
#print(count)

OBBViewer(ax, points)

#plot_implicit(ax, figure.f_rep, points)

plt.show()
"""



