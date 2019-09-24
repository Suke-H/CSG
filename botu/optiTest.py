import numpy as np
from scipy.optimize import minimize

from isNormalOK import sendData

#点群,法線,OBBの頂点を読み込み
points, normals, max_p, min_p = sendData()

#原点からの距離が最大の頂点を見つける
max_norm = 0
for v1, v2 in zip(max_p, min_p):
    if np.linalg.norm(v1) >= np.linalg.norm(v2):
        if np.linalg.norm(v1) >= max_norm:
            max_norm = np.linalg.norm(v1)
            
    else:
        if np.linalg.norm(v2) >= max_norm:
            max_norm = np.linalg.norm(v1)


def func(p, epsilon=0.7, alpha=np.pi/12):
    
    E = 0
    for pi, ni in zip(points, normals):
        #pと平面の距離(あとで二乗する)
        dist = (p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2]) / epsilon
        #平面の法線とpの法線との偏差(これも二乗する)
        theta = np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2]) / alpha

        E = E + np.exp(-dist**2) + np.exp(-theta**2)

    return -E

def func_deriv(p, epsilon=0.7, alpha=np.pi/12):

    dfdx = dfdy = dfdz = dfdd = 0

    for pi, ni in zip(points, normals):
        dfdx = dfdx + 2*pi[0]*(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])*np.exp(-(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])**2/epsilon**2)/epsilon**2 + 2*ni[0]*np.exp(-np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])**2/alpha**2)*np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])/(alpha**2*np.sqrt(1 - (p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])**2))
        dfdy = dfdy + 2*pi[1]*(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])*np.exp(-(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])**2/epsilon**2)/epsilon**2 + 2*ni[1]*np.exp(-np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])**2/alpha**2)*np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])/(alpha**2*np.sqrt(1 - (p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])**2))
        dfdz = dfdz + 2*pi[2]*(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])*np.exp(-(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])**2/epsilon**2)/epsilon**2 + 2*ni[2]*np.exp(-np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])**2/alpha**2)*np.arccos(p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])/(alpha**2*np.sqrt(1 - (p[0]*ni[0] + p[1]*ni[1] + p[2]*ni[2])**2))
    
        dfdd = dfdd -(2*p[3] - 2*p[0]*pi[0] - 2*p[1]*pi[1] - 2*p[2]*pi[2])*np.exp(-(p[3] - p[0]*pi[0] - p[1]*pi[1] - p[2]*pi[2])**2/epsilon**2)/epsilon**2
        
    return np.array([-dfdx, -dfdy, -dfdz, -dfdd])

cons = ({'type': 'eq',
         'fun' : lambda p: np.array([p[0]**2 + p[1]**2 + p[2]**2 - 1]),
         'jac' : lambda p: np.array([2*p[0], 2*p[1], 2*p[2], 0])},
        {'type': 'ineq',
         'fun' : lambda p: np.array([p[3] - max_norm]),
         'jac' : lambda p: np.array([0, 0, 0, 1])},
        {'type': 'ineq',
         'fun' : lambda p: np.array([-p[3]]),
         'jac' : lambda p: np.array([0, 0, 0, -1])}       
         )


result = minimize(func, [0,0,0,0], constraints=cons, method="SLSQP")




