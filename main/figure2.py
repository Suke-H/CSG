import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from f_rep2 import plot_implicit

def norm(normal):
     #ベクトルが一次元のとき
    if len(normal.shape)==1:
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
            
        else:
            return normal / np.linalg.norm(normal)

    #ベクトルが二次元
    else:
        #各法線のノルムをnormに格納
        norm = np.linalg.norm(normal, ord=2, axis=1)

        #normが0の要素は1にする(normalをnormで割る際に0除算を回避するため)
        norm = np.where(norm==0, 1, norm)

        #normalの各成分をノルムで割る
        norm = np.array([np.full(3, norm[i]) for i in range(len(norm))])
        return normal / norm



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

        #二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        #[[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

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
        
        #xが一次元のとき
        if type(x) is np.ndarray:
            #[[x,y,z],[x,y,z],...]のかたちにする
            normal = np.array([normal for i in range(X.shape[0])])

        return norm(normal)
      

class AND:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    #AND(f1,f2) = f1 + f2 - √f1^2 + f2^2
    def f_rep(self, x, y, z):
        return self.p1.f_rep(x,y,z) + self.p2.f_rep(x,y,z) - np.sqrt(self.p1.f_rep(x,y,z)**2 + self.p2.f_rep(x,y,z)**2)

    def normal(self, x, y, z):
        normal = self.p1.normal(x,y,z) + self.p2.normal(x,y,z)
        
        return norm(normal)


#f = z
p1 = plane([0, 0, 1, 0])
#f = 1.5 - z
p2 = plane([0, np.sqrt(2), np.sqrt(2), 1.5])

p3 = AND(p1, p2)

p4 = sphere([0, 0, 0, 1])
p5 = plane([-1,-1,-1,-1])

#plot_implicit(p3.f_rep)

X = np.array([1,0,0,0])
Y = np.array([0,1,0,0])
Z = np.array([0,0,1,0])

#print(p3.normal(0,0,1))
print(p3.normal(X,Y,Z))

