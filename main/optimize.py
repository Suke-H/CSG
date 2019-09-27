import numpy as np
from scipy.optimize import minimize

import open3d
from loadOBJ import loadOBJ
from OBB import buildOBB
from PreProcess import PreProcess
import figure as F

#####最適化する関数#################################################
###epsilon, alphaは調整が必要
def func(p, points, normals, fig, epsilon=0.7, alpha=np.pi/12):
	#print(points.shape, normals.shape, fig, epsilon, alpha)
	E = 0

	if fig == 0:
		figure = F.sphere(p)
	elif fig == 1:
		figure = F.plane(p)

	for xi, ni in zip(points, normals):
        #pと平面の距離(あとで二乗する)
		dist = figure.f_rep(xi) / epsilon
        #平面の法線とpの法線との偏差:0と180度がmax(これも二乗する)
		theta = np.arccos(abs(np.dot(figure.normal(xi), ni))) / alpha
		E = E + np.exp(-dist**2) + np.exp(-theta**2)

		#sys.stdout.write("\r p{}\n xi{}\n ni{}".format(p, xi, ni))
		#time.sleep(0.1)

	#print("p:{}\n E:{}\n".format(p, E), file=codecs.open('test.txt', 'a', 'utf-8'))
    #print(E)

    #最小化なのでマイナスを返す
	return -E

#####最適化#####################################################
def figOptimize(points, normals, length, fig):

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

	#定数(数？)
	arg = (points, normals, fig, 0.7*length, np.pi/12)

    #最適化
	result = minimize(func, p_0, args=arg, constraints=cons, method='SLSQP')

	return result

#print(figOptimize("../data/pumpkin.obj", 1))
