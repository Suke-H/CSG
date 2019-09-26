import numpy as np

class sphere:
    def __init__(self, p):
        self.p = p

	#球の関数:f = r - √(x-x0)^2+(y-y0)^2+(z-z0)^2
    def f_rep(self, xi):
        r0 = np.asarray([self.p[i] for i in range(3)])
        return self.p[3] - np.linalg.norm(xi-r0)
        
    def f_rep_draw(self, x, y, z):
        return self.p[3] - np.sqrt((x-self.p[0])**2 + (y-self.p[1])**2 + (z-self.p[2])**2)


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
        
    def f_rep_draw(self, x, y, z):
        return self.p[3] - (self.p[0]*x + self.p[1]*y + self.p[2]*z) 

	#平面の法線:[n1, n2, n3]#すでに単位ベクトルだが一応#xiいらない
    def normal(self, xi):
        normal = np.asarray([self.p[i] for i in range(3)])
        if np.linalg.norm(normal) == 0:
			#print("Warning: 法線ベクトルがゼロです！")
            return normal
            
        else:
            return normal / np.linalg.norm(normal)
