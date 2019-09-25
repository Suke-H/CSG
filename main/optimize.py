import numpy as np
from scipy.optimize import minimize

import open3d
from loadOBJ import loadOBJ
from OBB import buildOBB
from PreProcess import PreProcess

##########図形の設定##########################################
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

	#平面の法線:[n1, n2, n3]#すでに単位ベクトルだが一応#xiいらない
	def normal(self, xi):
		normal = np.asarray([self.p[i] for i in range(3)])
		if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
			return normal

		else:
			return normal / np.linalg.norm(normal)


#####最適化する関数#################################################
###epsilon, alphaは調整が必要
def func(p, points, normals, fig, epsilon=0.7, alpha=np.pi/12):
	#print(points.shape, normals.shape, fig, epsilon, alpha)
	E = 0

	if fig == 0:
		figure = sphere(p)
	elif fig == 1:
		figure = plane(p)

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
def figOptimize(path, fig=0):

	#点群,法線,OBBの対角線の長さ  取得
	points, normals, length = PreProcess(path)

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

print(figOptimize("../data/pumpkin.obj"))
