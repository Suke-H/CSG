import numpy as np
import numpy.linalg as LA
import itertools

"""
def line(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    z = a[2]*t + b[2]*(1-t)

    
    p = a*t + b*(1-t)
    #xyzに分解
    N = p.T[:]
    x = N[0, :]
    y = N[1, :]
    z = N[2, :]
    
"""

def buildOBB(points):
    #分散共分散行列Sを生成
    S = np.cov(points, rowvar=0, bias=1)

    #固有ベクトルを算出
    w,svd_vector = LA.eig(S)
    #sorted_svd_index = np.argsort(w)
    #svd_vector = v[sorted_svd_index]

    #print(S)
    #print(svd_vector)
    #print("="*50)

    #正規直交座標にする(=直行行列にする)
    #############################################
    u = np.asarray([svd_vector[i] / np.linalg.norm(svd_vector[i]) for i in range(3)])

    #print(u)
    #print("="*50)

    #点群の各点と各固有ベクトルとの内積を取る
    #P V^T = [[p1*v1, p1*v2, p1*v3], ... ,[pN*v1, pN*v2, pN*v3]]
    inner_product = np.dot(points, u.T)
    
    #各固有値の内積最大、最小を抽出(max_stu_point = [s座標max, tmax, umax])
    max_stu_point = np.amax(inner_product, axis=0)
    min_stu_point = np.amin(inner_product, axis=0)

    #xyz座標に変換・・・単位ベクトル*座標
    #max_xyz_point = [[xs, ys, zs], [xt, yt, zt], [xu, yu, zu]]
    max_xyz_point = np.asarray([u[i]*max_stu_point[i] for i in range(3)])
    min_xyz_point = np.asarray([u[i]*min_stu_point[i] for i in range(3)])

    """
    max_index = 
    print(max_index)
    max_point = np.asarray([points[max_index[i]] for i in range(3)])

    min_index = np.argmin(inner_product, axis=0)
    min_point = np.asarray([points[min_index[i]] for i in range(3)])
    """
    #対角線の長さ
    vert_max = min_xyz_point[0] + min_xyz_point[1] + max_xyz_point[2]
    vert_min = max_xyz_point[0] + max_xyz_point[1] + min_xyz_point[2]
    l = np.linalg.norm(vert_max-vert_min)

    return max_xyz_point, min_xyz_point, l

def buildAABB(points):
    #なんとこれで終わり
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    return max_p, min_p