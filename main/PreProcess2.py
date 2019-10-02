import numpy as np
from scipy.optimize import minimize

import open3d
from loadOBJ import loadOBJ
from OBB import buildOBB
from f_rep2 import MakePoints, cube, p_z0

def PreProcess2():
	#自作した点群を読み込み
    points, X, Y, Z = MakePoints(cube)

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
