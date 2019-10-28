import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from method import *


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
        if type(x) is np.ndarray or type(x) is list:
            #[[x,y,z],[x,y,z],...]のかたちにする
            normal = np.array([normal for i in range(x.shape[0])])

        return norm(normal)

class cylinder:
    def __init__(self, p):
        #|(p-p0) × v| = r
        # (p0 = [x0, y0, z0], v = [a, b, c])
        # para = [x0, y0, z0, a, b, c, r]
        self.x0 = p[0]
        self.y0 = p[1]
        self.z0 = p[2]
        self.a = p[3]
        self.b = p[4]
        self.c = p[5]
        self.r = p[6]

	#円柱の関数
    def f_rep(self, x, y, z):
        return self.r - np.sqrt(( self.b*(z-self.z0)-self.c*(y-self.y0))**2 + \
            (self.c*(x-self.x0)-self.a*(z-self.z0))**2  + (self.a*(y-self.y0)-self.b*(x-self.x0))**2 )

	#円柱の法線
    def normal(self, x, y, z):
        normal = np.array([self.c*(self.c*(x-self.x0)-self.a*(z-self.z0)) - self.b*(self.a*(y-self.y0)-self.b*(x-self.x0)), \
            -self.c*(self.b*(z-self.z0)-self.c*(y-self.y0)) + self.a*(self.a*(y-self.y0)-self.b*(x-self.x0)), \
            self.b*(self.b*(z-self.z0)-self.c*(y-self.y0)) - self.a*(self.c*(x-self.x0)-self.a*(z-self.z0))])

        #二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        #[[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

class cone:
    def __init__(self, p):
        #(p-p0)・v = |p-p0||v|cosθ
        # (p0 = [x0, y0, z0], v = [a, b, c])
        # para = [x0, y0, z0, a, b, c, θ]
        self.x0 = p[0]
        self.y0 = p[1]
        self.z0 = p[2]
        self.a = p[3]
        self.b = p[4]
        self.c = p[5]
        self.t = p[6]

    #円錐の関数
    def f_rep(self, x, y, z):
        return np.cos(self.t) - (self.a*(x-self.x0)+self.b*(y-self.y0)+self.c*(z-self.z0)) / \
            np.sqrt(((x-self.x0)**2+(y-self.y0)**2+(z-self.z0)**2) * (self.a**2+self.b**2+self.c**2))

    #円錐の法線
    def normal(self, x, y, z):
        normal = np.array([self.a*(self.a*(x-self.x0)+self.b*(y-self.y0)+self.c*(z-self.z0)) \
            -(self.a**2+self.b**2+self.c**2)*np.cos(self.t)**2*(x-self.x0), \
            self.b*(self.a*(x-self.x0)+self.b*(y-self.y0)+self.c*(z-self.z0)) \
            -(self.a**2+self.b**2+self.c**2)*np.cos(self.t)**2*(y-self.y0), \
            self.c*(self.a*(x-self.x0)+self.b*(y-self.y0)+self.c*(z-self.z0)) \
            -(self.a**2+self.b**2+self.c**2)*np.cos(self.t)**2*(z-self.z0)])

        #二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        #[[x1,y1,z1],...]の形にする
        normal = normal.T
        
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

