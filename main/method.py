import numpy as np
import numpy.linalg as LA
import itertools

#pointsからpのk近傍点のindexのリストを返す
def K_neighbor(points, p, k):
    #points[i]とpointsの各点とのユークリッド距離を格納
    distances = np.sum(np.square(points - p), axis=1)

    #距離順でpointsをソートしたときのインデックスを格納
    sorted_index = np.argsort(distances)

    return sorted_index[:k]

#点群データなどをx, y, zに分解する

#[x1, y1, z1]         [x1, x2, ..., xn]
#      :        ->    [y1, y2, ..., yn]
#[xn, yn, zn]         [z1, z2, ..., zn]
def Disassemble(XYZ):
    XYZ = XYZ.T[:]
    X = XYZ[0, :]
    Y = XYZ[1, :]
    Z = XYZ[2, :]

    return X, Y, Z

def line(a, b):
    t = np.arange(0, 1, 0.01)

    x = a[0]*t + b[0]*(1-t)
    y = a[1]*t + b[1]*(1-t)
    z = a[2]*t + b[2]*(1-t)

    return x, y, z

###OBB生成####
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

###AABB生成####
def buildAABB(points):
    #なんとこれで終わり
    max_p = np.amax(points, axis=0)
    min_p = np.amin(points, axis=0)

    return max_p, min_p

#点群を入力としてOBBを描画する
def OBBViewer(ax, points):
    #OBB生成
    max_p, min_p, _ = buildOBB(points)

    #直積：[smax, smin]*[tmax, tmin]*[umax, umin] <=> 頂点
    s_axis = np.vstack((max_p[0], min_p[0]))
    t_axis = np.vstack((max_p[1], min_p[1]))
    u_axis = np.vstack((max_p[2], min_p[2]))

    products = np.asarray(list(itertools.product(s_axis, t_axis, u_axis)))
    vertices = np.sum(products, axis=1)

    #各頂点に対応するビットの列を作成
    bit = np.asarray([1, -1])
    vertices_bit = np.asarray(list(itertools.product(bit, bit, bit)))


    #頂点同士のハミング距離が1なら辺を引く
    for i, v1 in enumerate(vertices_bit):
        for j, v2 in enumerate(vertices_bit):
            if np.count_nonzero(v1-v2) == 1:
                x, y, z = line(vertices[i], vertices[j])
                ax.plot(x,y,z,marker=".",color="orange")

    #OBBの頂点の1つ
    vert_max = min_p[0] + min_p[1] + max_p[2]
    vert_min = max_p[0] + max_p[1] + min_p[2]

    #xyzに分解
    Xmax, Ymax, Zmax = Disassemble(max_p)
    Xmin, Ymin, Zmin = Disassemble(min_p)

    #頂点なども描画
    ax.plot(Xmax,Ymax,Zmax,marker="X",linestyle="None",color="red")
    ax.plot(Xmin,Ymin,Zmin,marker="X",linestyle="None",color="blue")
    ax.plot([vert_max[0], vert_min[0]],[vert_max[1], vert_min[1]],[vert_max[2], vert_min[2]],marker="o",linestyle="None",color="black")

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