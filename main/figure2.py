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
        
        #[[x,y,z],[x,y,z],...]のかたちにする
        normal = np.array([normal for i in range(x.shape[0])])

        return norm(normal)

class cylinder:
    def __init__(self, p):
        #|(p-p0) × v| = r
        # (p0 = [x0, y0, z0], v = [a, b, c])
        # para = [x0, y0, z0, a, b, c, r]
        self.p = p

	#円柱の関数
    def f_rep(self, x, y, z):
        (x0, y0, z0, a, b, c, r) = self.p
        return r - np.sqrt(( b*(z-z0)-c*(y-y0))**2 + \
            (c*(x-x0)-a*(z-z0))**2  + (a*(y-y0)-b*(x-x0))**2 )

	#円柱の法線
    def normal(self, x, y, z):
        (x0, y0, z0, a, b, c, r) = self.p
        normal = np.array([c*(c*(x-x0)-a*(z-z0)) - b*(a*(y-y0)-b*(x-x0)), \
            -c*(b*(z-z0)-c*(y-y0)) + a*(a*(y-y0)-b*(x-x0)), \
            b*(b*(z-z0)-c*(y-y0)) - a*(c*(x-x0)-a*(z-z0))])

        #二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        #[[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

class cone:
    def __init__(self, p):
        #(p-p0)・v = |p-p0||v|cosθ
        # (p0 = [x0, y0, z0], v = [a, b, c])
        # para = [x0, y0, z0, a, b, c, θ]
        self.p = p

    #円錐の関数
    def f_rep(self, x, y, z):
        (x0, y0, z0, a, b, c, t) = self.p
        return np.cos(t) - (a*(x-x0)+b*(y-y0)+c*(z-z0)) / \
            np.sqrt(((x-x0)**2+(y-y0)**2+(z-z0)**2) * (a**2+b**2+c**2))

    #円錐の法線
    def normal(self, x, y, z):
        (x0, y0, z0, a, b, c, t) = self.p
        normal = np.array([a*(a*(x-x0)+b*(y-y0)+c*(z-z0)) - (a**2+b**2+c**2)*np.cos(t)**2*(x-x0), \
            b*(a*(x-x0)+b*(y-y0)+c*(z-z0)) - (a**2+b**2+c**2)*np.cos(t)**2*(y-y0), \
            c*(a*(x-x0)+b*(y-y0)+c*(z-z0)) - (a**2+b**2+c**2)*np.cos(t)**2*(z-z0)])

        #二次元のとき[[x1,...],[y1,...][z1,...]]と入っているため
        #[[x1,y1,z1],...]の形にする
        normal = normal.T
        
        return norm(normal)

class AND:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    # AND(f1,f2) = f1 + f2 - √f1^2 + f2^2
    def f_rep(self, x, y, z):
        return self.p1.f_rep(x,y,z) + self.p2.f_rep(x,y,z) - np.sqrt(self.p1.f_rep(x,y,z)**2 + self.p2.f_rep(x,y,z)**2)

    # ∇AND = ∇f1 + ∇f2 - (f1∇f1+f2∇f2)/√f1^2+f2^2
    def normal(self, x, y, z):
        f1, f2, n1, n2 = self.p1.f_rep(x,y,z), self.p2.f_rep(x,y,z), self.p1.normal(x,y,z), self.p2.normal(x,y,z)
        #normal = n1+n2
        normal = np.array([n1[i] + n2[i] - (f1[i]*n1[i] + f2[i]*n2[i]) / np.sqrt(f1[i]**2 + f2[i]**2) for i in range(len(f1))])
        
        return norm(normal)

class OR:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    #OR(f1,f2) = f1 + f2 + √f1^2 + f2^2
    def f_rep(self, x, y, z):
        return self.p1.f_rep(x,y,z) + self.p2.f_rep(x,y,z) + np.sqrt(self.p1.f_rep(x,y,z)**2 + self.p2.f_rep(x,y,z)**2)

    #∇OR = ∇f1 + ∇f2 + (f1∇f1+f2∇f2)/√f1^2+f2^2
    def normal(self, x, y, z):
        f1, f2, n1, n2 = self.p1.f_rep(x,y,z), self.p2.f_rep(x,y,z), self.p1.normal(x,y,z), self.p2.normal(x,y,z)
        #normal = n1+n2
        normal = np.array([n1[i] + n2[i] + (f1[i]*n1[i] + f2[i]*n2[i]) / np.sqrt(f1[i]**2 + f2[i]**2) for i in range(len(f1))])
        
        return norm(normal)

class NOT:
    def __init__(self, p):
        self.p = p

    #NOT(f) = -f
    def f_rep(self, x, y, z):
        return -self.p.f_rep(x,y,z)

    #∇NOT = -∇f
    def normal(self, x, y, z):
        normal = -self.p.normal(x,y,z)
        
        return norm(normal)

