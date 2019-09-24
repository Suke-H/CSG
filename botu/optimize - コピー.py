import numpy as np
from scipy.optimize import minimize

import open3d
from loadOBJ import loadOBJ
from OBB import buildOBB

import sys

#球：０、平面：１、円柱：２
fig = 0

C = True
count1 = count2 = 0

#################点群と法線準備#########################

#objファイルをnumpyで読み込んで,open3dのデータ形式に
points, _, _, _ = loadOBJ("./data/teapot.obj")
pointcloud = open3d.PointCloud()
pointcloud.points = open3d.Vector3dVector(points)

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
normals = np.asarray(pointcloud.normals)

###########OBB生成###############################################
#(最適化の条件にも使いたい)
_, _, length = buildOBB(points)

##########図形の関数##########################################
class sphere:
	def __init__(self, p):
		self.p = p

	#球の関数:f = r - √(x-x0)^2+(y-y0)^2+(z-z0)^2
	def f_rep(self, xi):
		r0 = np.asarray([self.p[i] for i in range(3)])
		return self.p[3] - np.linalg.norm(xi-r0)


	#球の法線:[2(x-x0), 2(y-y0), 2(z-z0)]※単位ベクトルを返す
	def normal(self, xi):
		normal = np.asarray([2*(self.p[i]-xi[i]) for i in range(3)])
		if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
			return normal
			
		else:
			return normal / np.linalg.norm(normal)

class plane:
	def __init__(self, p):
		self.p = p

	#平面の関数:f = d - (n1*x+n2*y+n3*z)
	def f_rep(self, xi):
		n = np.asarray([self.p[i] for i in range(3)])
		return self.p[3] - np.dot(n, xi)

	#平面の法線:[n1, n2, n3]#すでに単位ベクトル#xiいらない
	def normal(self, xi):
		return np.asarray([self.p[i] for i in range(3)])

#####最適化する関数#################################################
###epsilon, alphaは調整が必要
def func(p, epsilon=0.7*length, alpha=np.pi/12):
	E = 0
	global C
	global count1
	if fig == 0:
		figure = sphere(p)
	elif fig == 1:
		figure = plane(p)

	with open(path_w, mode='w') as f:
    	f.write(str(count1) + "\n")

	for xi, ni in zip(points, normals):

		with open(path_w, mode='w') as f:
    		f.write(str(count2) + "\n")


        #pと平面の距離(あとで二乗する)
		dist = figure.f_rep(xi) / epsilon
        #平面の法線とpの法線との偏差:0と180度がmax(これも二乗する)
		#if np.isnan(abs(np.dot(figure.normal(xi), ni))) and C:
			#print(p, ni, count)
			#print(dist)
			#C = False
		theta = np.arccos(abs(np.dot(figure.normal(xi), ni))) / alpha
		E = E + np.exp(-dist**2) + np.exp(-theta**2)
		np.savetxt('sample.txt', xi)

	#if np.isnan(E) and C:
		#print(p, xi, ni, count)
		#C = False

	count = count + 1
        
    #print(E)

    #最小化なのでマイナスを返す
	return -E

#####最適化#####################################################
def figOptimize(fig_type=0):
	global fig
	fig = fig_type

	####条件。eqは=, ineqは>=####
	#球の条件
	if fig == 0:
		print("球の条件")
		cons = (
        {'type': 'ineq',
         'fun' : lambda p: np.array([p[3]])}
        )

		p_0 = [0.1, 0.1, 0.1, 0.1]

	#平面の条件
	if fig == 1:
		print("平面の条件")
		cons = (
        #{'type': 'eq',
        # 'fun' : lambda p: np.array([p[0]**2 + p[1]**2 + p[2]**2 - 1])}
        )
    
		p_0 = [1, 0, 0, 0.1]
	

    #最適化
	result = minimize(func, p_0, constraints=cons, method='SLSQP')
	
	return result

print(figOptimize(0))
