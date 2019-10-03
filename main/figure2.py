import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from f_rep2 import plot_implicit

class sphere:
    def __init__(self, p):
        #パラメータ
        self.p = p

    #球の方程式: f(x,y,z) = r - √(x-a)^2 + (y-b)^2 + (z-c)^2
    def f_rep(self, x, y, z):
        return self.p[3] - np.sqrt((x-self.p[0])**2 + (y-self.p[1])**2 + (z-self.p[2])**2)

	#球の法線:[2(x-x0), 2(y-y0), 2(z-z0)]※単位ベクトルを返す
    def normal(self, x, y, z):
        normal = np.array([x-self.p[0], y-self.p[1], z-self.p[2]])
        
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
        
        else:
            return normal / np.linalg.norm(normal)

class plane:
    def __init__(self, p):
        self.p = p

	#平面の関数:f = d - (n1*x+n2*y+n3*z)
    def f_rep(self, x, y, z):
        return self.p[3] - (self.p[0]*x + self.p[1]*y + self.p[2]*z)

	#平面の法線:[n1, n2, n3]#すでに単位ベクトルだが一応
    #x, y, zは実質いらない
    def normal(self, x, y, z):
        normal = np.array([self.p[0], self.p[1], self.p[2]])
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
            
        else:
            return normal / np.linalg.norm(normal)

class AND:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    #AND(f1,f2) = f1 + f2 - √f1^2 + f2^2
    def f_rep(self, x, y, z):
        return self.p1.f_rep(x,y,z) + self.p2.f_rep(x,y,z) - np.sqrt(self.p1.f_rep(x,y,z)**2 + self.p2.f_rep(x,y,z)**2)

    def normal(self, x, y, z):
        normal = self.p1.normal(x,y,z) + self.p2.normal(x,y,z)
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
            
        else:
            return normal / np.linalg.norm(normal)


#f = z
p1 = plane([0, 0, 1, 0])
#f = 1.5 - z
p2 = plane([0, 0, -1, 1.5])

p3 = AND(p1, p2)

plot_implicit(p3.f_rep)

