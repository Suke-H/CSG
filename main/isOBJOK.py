import numpy as np
import open3d

from loadOBJ import loadOBJ

##OBJファイルをopen3dの形式に変換して可視化できるか##

#OBJファイルを読み込み
vertices, X, Y, Z = loadOBJ("./pumpkin.obj")
pointcloud = open3d.PointCloud()
pointcloud.points = open3d.Vector3dVector(vertices)

# 可視化
open3d.draw_geometries([pointcloud])
