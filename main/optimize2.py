import numpy as np
from scipy.optimize import minimize

import open3d
from loadOBJ import loadOBJ
from OBB import buildOBB
from PreProcess import PreProcess
import figure2 as F

#####最適化する関数#################################################
###epsilon, alphaは調整が必要
def func(p, X, Y, Z, normals, fig, epsilon=0.7, alpha=np.pi/12):
	#print(points.shape, normals.shape, fig, epsilon, alpha)
	E = 0

	if fig == 0:
		figure = F.sphere(p)
	elif fig == 1:
		figure = F.plane(p)

    #dist[i] = i番目の点からの垂直距離
	dist = figure.f_rep(X,Y,Z) / epsilon

	#theta[i] = i番目の点の法線とnormalとの偏差(角度の差)
	#np.sumは各点の法線同士の内積を取っている
	#[nf_1*ni_1, nf_2*ni_2, ...]みたいな感じ
	theta = np.arccos(np.abs(np.sum(figure.normal(X,Y,Z) * normals, axis=1))) / alpha

	E = np.sum(np.exp(-dist**2) + np.exp(-theta**2))

    #最小化なのでマイナスを返す
	return -E

#####最適化#####################################################
def figOptimize2(X, Y, Z, normals, length, fig):

	####条件###################
	# eqは=, ineqは>=

	#球の条件
	if fig == 0:
		print("球")
		cons = (
        {'type': 'ineq',
         'fun' : lambda p: np.array([p[3]])}
        )

		p_0 = [0.1, 0.1, 0.1, 0.1]

	#平面の条件
	if fig == 1:
		print("平面")
		cons = (
        {'type': 'eq',
         'fun' : lambda p: np.array([p[0]**2 + p[1]**2 + p[2]**2 - 1])}
        )

		p_0 = [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3), 0.1]

	#定数(pをのぞく引数)
	arg = (X, Y, Z, normals, fig, 0.7*length, np.pi/12)

    #最適化
	result = minimize(func, p_0, args=arg, constraints=cons, method='SLSQP')

	return result

#print(figOptimize("../data/pumpkin.obj", 1))
