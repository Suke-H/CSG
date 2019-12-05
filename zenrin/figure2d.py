import numpy as np

class circle:
    def __init__(self, p):
        # パラメータ
        # p = [x0, y0, r]
        self.p = p

    # 円の方程式: f(x,y) = r - √(x-x0)^2 + (y-y0)^2
    def f_rep(self, x, y):
        x0, y0, r = self.p

        return r - np.sqrt((x-x0)**2 + (y-y0)**2)

class line:
    def __init__(self, p):
        # パラメータ
        # p = [a, b, c]
        self.p = p

    # 直線の方程式: f(x,y) = c - (ax + by)
    def f_rep(self, x, y):
        a, b, c = self.p

        return c - (a*x + b*y)

class rectangle:
    def __init__(self, p):
        # パラメータ
        # p = [x0, y0, w, h, t]
        # x0, y0: 中心
        # w, h: 幅、高さ
        # t: 中心から反時計回りへの回転の角度(/rad)
        self.p = p

    # 長方形: spin(inter(l1,l2,l3,l4), x0, y0, t)
    def f_rep(self, x, y):
        x0, y0, w, h, t = self.p
        # 4辺作成
        l1 = line([0,1,y0+h/2])
        l2 = line([0,-1,-y0+h/2])
        l3 = line([-1,0,-x0+w/2])
        l4 = line([1,0,x0+w/2])
        # intersectionで長方形作成
        rect = inter(l1, inter(l2, inter(l3, l4)))
        # 回転
        rect = spin(rect, x0, y0, t)

        return rect.f_rep(x, y)


class spin:
    def __init__(self, fig, x0, y0, t):
        #fig: 回転させる図形
        self.fig = fig
        #x0: 回転の中心
        self.x0 = np.array([x0, y0])
        # t: 点x0を中心に反時計回りに回転させる角度(/rad)
        self.t = t
        # P: 回転行列
        self.P = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        # P_inv: Pの逆行列
        self.P_inv = np.linalg.inv(self.P)

    # X = P(x-x0) + x0     で回転させることができるため、
    # x = P_inv(X-x0) + x0 をf(x)に代入する
    def f_rep(self, x, y):
        # xとyをまとめて[[x1,x2,...],[y1,y2,...]]の形にする
        p0 = np.concatenate([[x],[y]])
        
        # x0を[[x0,x0,...],[y0,y0,...]]にする
        x0 = np.array([[self.x0[0] for i in range(len(x))], [self.x0[1] for i in range(len(x))]])

        # x = P_inv(X-x0) + x0
        x, y = np.dot(self.P_inv, p0-x0) + x0

        # f(x)に代入
        return self.fig.f_rep(x, y)

# intersection(=論理積, and)
class inter:
    def __init__(self, fig1, fig2):
        self.fig1 = fig1
        self.fig2 = fig2

    # inter(f1,f2) = f1 + f2 - √f1^2 + f2^2
    def f_rep(self, x, y):
        f1 = self.fig1.f_rep(x,y)
        f2 = self.fig2.f_rep(x,y)

        return f1 + f2 - np.sqrt(f1**2 + f2**2)

# union(=論理和, or)
class union:
    def __init__(self, fig1, fig2):
        self.fig1 = fig1
        self.fig2 = fig2

    # union(f1,f2) = f1 + f2 + √f1^2 + f2^2
    def f_rep(self, x, y):
        f1 = self.fig1.f_rep(x,y)
        f2 = self.fig2.f_rep(x,y)
        
        return f1 + f2 + np.sqrt(f1**2 + f2**2)

    