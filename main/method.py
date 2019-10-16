import numpy as np
from OBB import buildAABB

#pointsからpのk近傍点のindexのリストを返す
def K_neighbor(points, p, k):
    #points[i]とpointsの各点とのユークリッド距離を格納
    distances = np.sum(np.square(points - p), axis=1)

    #距離順でpointsをソートしたときのインデックスを格納
    sorted_index = np.argsort(distances)

    return sorted_index[:k]

#陰関数のグラフ描画
#fn  ...fn(x, y, z) = 0の左辺
#AABB_size ...AABBの各辺をAABB_size倍する
def plot_implicit(ax, fn, points, AABB_size=2, contourNum=30):
    #AABB生成
    max_p, min_p = buildAABB(points)

    xmax, ymax, zmax = max_p[0], max_p[1], max_p[2]
    xmin, ymin, zmin = min_p[0], min_p[1], min_p[2]

    #AABBの各辺がAABB_size倍されるように頂点を変更
    xmax = xmax + (xmax - xmin)/2 * AABB_size
    xmin = xmin - (xmax - xmin)/2 * AABB_size
    ymax = ymax + (ymax - ymin)/2 * AABB_size
    ymin = ymin - (ymax - ymin)/2 * AABB_size
    zmax = zmax + (zmax - zmin)/2 * AABB_size
    zmin = zmin - (zmax - zmin)/2 * AABB_size

    A_X = np.linspace(xmin, xmax, 100) # resolution of the contour
    A_Y = np.linspace(ymin, ymax, 100)
    A_Z = np.linspace(zmin, zmax, 100)
    B_X = np.linspace(xmin, xmax, 15) # number of slices
    B_Y = np.linspace(ymin, ymax, 15)
    B_Z = np.linspace(zmin, zmax, 15)
    #A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B_Z: # plot contours in the XY plane
        X,Y = np.meshgrid(A_X, A_Y)
        Z = fn(X,Y,z)
        ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B_Y: # plot contours in the XZ plane
        X,Z = np.meshgrid(A_X, A_Z)
        Y = fn(X,y,Z)
        ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B_X: # plot contours in the YZ plane
        Y,Z = np.meshgrid(A_Y, A_Z)
        X = fn(x,Y,Z)
        ax.contour(X+x, Y, Z, [x], zdir='x')

    #(拡大した)AABBの範囲に制限
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)