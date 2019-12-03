import numpy as np
from scipy.optimize import minimize

import open3d
from loadOBJ import loadOBJ
from method import *
from PreProcess import PreProcess
import figure2 as F

E_list = []



def RandomInit(fig):
	###最適化の条件に必要な道具###

	#単位ベクトルn
	n = np.array([0, 0, 0])
	while np.linalg.norm(n)==0:
		print("!!")
		n = np.random.rand(3)
	
	n = n / np.linalg.norm(n)

	#正数r
	r = np.abs(np.random.rand())

	#5°<θ<80°
	a, b = np.pi/36, np.pi/(180/80)
	t = (b - a) * np.random.rand() + a
	############################

	#球：r > 0
	if fig == 0:
		p = np.random.rand(3)
		p = np.append(p, r)

	#平面：|n| = 1
	elif fig == 1:
		p = np.append(n, np.random.rand())

	#円柱：|v| = 1, r
	elif fig == 2:
		p = np.random.rand(3)
		p = np.append(p, n)
		p = np.append(p, r)

	else:
		p = np.random.rand(3)
		p = np.append(p, n)
		p = np.append(p, t)

	return p




#####最適化する関数#################################################
###epsilon, alphaは調整が必要
def func(p, X, Y, Z, normals, fig, epsilon=0.7, alpha=np.pi/12):
	#print(points.shape, normals.shape, fig, epsilon, alpha)
	E = 0

	if fig == 0:
		figure = F.sphere(p)
	elif fig == 1:
		figure = F.plane(p)
	elif fig==2:
		figure = F.cylinder(p)
	else:
		figure = F.cone(p)

    #dist[i] = i番目の点からの垂直距離
	dist = figure.f_rep(X,Y,Z) / epsilon

	#theta[i] = i番目の点の法線とnormalとの偏差(角度の差)
	#np.sumは各点の法線同士の内積を取っている
	#[nf_1*ni_1, nf_2*ni_2, ...]みたいな感じ
	theta = np.arccos(np.abs(np.sum(figure.normal(X,Y,Z) * normals, axis=1))) / alpha

	E = np.sum(np.exp(-dist**2) +  np.exp(-theta**2))

    #最小化なのでマイナスを返す

	global E_list
	E_list.append(E)
	return -E

#####最適化#####################################################
def figOptimize2(X, Y, Z, normals, length, fig):

	####条件###################
	# eqは=, ineqは>=

	#球の条件
	if fig == 0:
		print("球")

		# 0 < r < 5*length
		cons = (
        {'type': 'ineq',
         'fun' : lambda p: np.array([p[3]])},
		{'type': 'ineq',
         'fun' : lambda p: np.array([5*length - p[3]])}

        )

		#定数(pをのぞく引数)
		arg = (X, Y, Z, normals, fig, 0.01*length, np.pi/12)


	#平面の条件
	if fig == 1:
		print("平面")

		#|n| = 1
		cons = (
        {'type': 'eq',
         'fun' : lambda p: np.array([p[0]**2 + p[1]**2 + p[2]**2 - 1])}
        )

		#定数(pをのぞく引数)
		arg = (X, Y, Z, normals, fig, 0.07*length, np.pi/12)

	#円柱の条件
	if fig == 2:
		print("円柱")

		#|v| = 1
		#0 < r
		cons = (
        {'type': 'eq',
         'fun' : lambda p: np.array([p[3]**2 + p[4]**2 + p[5]**2 - 1])}, 
		{'type': 'ineq',
         'fun' : lambda p: np.array([p[6]])}
        )

		#定数(pをのぞく引数)
		arg = (X, Y, Z, normals, fig, 0.05*length, np.pi/12)

	#円錐の条件
	if fig == 3:
		print("円錐")

		#|v| = 1
		#0 < θ < π/2
		cons = (
        {'type': 'eq',
         'fun' : lambda p: np.array([p[3]**2 + p[4]**2 + p[5]**2 - 1])}, 
		{'type': 'ineq',
         'fun' : lambda p: np.array([p[6]-np.pi/(180/10)])},
		{'type': 'ineq',
         'fun' : lambda p: np.array([-p[6]+np.pi/(180/60)])}
        )

		#定数(pをのぞく引数)
		arg = (X, Y, Z, normals, fig, 0.03*length, np.pi/9)

	p_0 = RandomInit(fig)

    #最適化
	result = minimize(func, p_0, args=arg, constraints=cons, method='SLSQP')

	global E_list
	#print(E_list)

	return result

#print(figOptimize("../data/pumpkin.obj", 1))
