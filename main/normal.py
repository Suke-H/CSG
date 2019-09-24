import numpy as np
import numpy.linalg as LA

def normalEstimate(points,  K=5):
    N = points.shape[0]
    normal = []

    for p in points:

        ###K個の近傍点を取り出す(K=Nならpointsすべて使う)###
        if K!=N:

            #pとpointsの各点とのユークリッド距離を格納
            distances = np.sum(np.square(points - p), axis=1)

            #距離順でpointsをソートし,K個取り出す
            sorted_index = np.argsort(distances)
            sorted_points = points[sorted_index]
            #print(distances)
            #print(sorted_index)
            K_neighbor = sorted_points[:K, :]
            #print(K_neighbor)
        
        else:
            K_neighbor = points[:]

        ###共分散行列###
        S = np.cov(K_neighbor, rowvar=0, bias=1)

        ###固有ベクトル###
        w,v = LA.eig(S)
        sorted_svd_index = np.argsort(w)
        sorted_svd_vector = v[sorted_svd_index]
        
        #最小の固有値に対応する固有ベクトルが法線
        normal.append(sorted_svd_vector[0])

    return np.asarray(normal)




        

P = np.asarray([[1,2,3],[4,5,6],[7,8,9], [3, 4, 1], [2, 1, 3], [6, 1, 2]])
print(normalEstimate(P))

