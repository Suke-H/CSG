import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

from loadOBJ import loadOBJ
import figure2 as F
from optimize import figOptimize
from optimize2 import figOptimize2
from PreProcess import PreProcess
from PreProcess2 import PreProcess2
from method import *

#seabornはimportしておくだけでもmatplotlibのグラフがきれいになる
import seaborn as sns
sns.set_style("darkgrid")


def sh(x, y, z):
        return 37.29042182 - np.sqrt((x+3.10735045)**2 + (y-1.81359686)**2 + (z+110.75950196)**2)

#mainの部分
def OptiViewer(path, fig_type):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #点群,法線,OBBの対角線の長さ  取得
    #points, X, Y, Z, normals, length = PreProcess(path)
    
    #自作の点群を扱いたいときはこちら
    points, X, Y, Z, normals, length = PreProcess2()

    print("points:{}".format(points.shape[0]))

    #点群を描画
    ax.plot(X,Y,Z,marker=".",linestyle='None',color="green")

    U, V, W = Disassemble(normals)

    #法線を描画
    #ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True)

    #OBBを描画
    OBBViewer(ax, points)

    ###最適化###
    #result = figOptimize(points, normals, length, fig_type)
    result = figOptimize2(X, Y, Z, normals, length, fig_type)
    print(result)

    #fig_typeに応じた図形を選択
    if fig_type==0:
        figure = F.sphere(result.x)
    elif fig_type==1:
        figure = F.plane(result.x)

    #最適化された図形を描画
    plot_implicit(ax, figure.f_rep, points, AABB_size=1, contourNum=15)

    #最後に.show()を書いてグラフ表示
    plt.show()

"""
def DetectViewer(path):
    

    #点群,法線,OBBの対角線の長さ  取得
    #points, X, Y, Z, normals, length = PreProcess(path)
    
    #自作の点群を扱いたいときはこちら
    points, X, Y, Z, normals, length = PreProcess2()

    print("points:{}".format(points.shape[0]))
 
    score = []
    para = []

    ###最適化###
    for fig_type in [0, 1]:

        ###グラフ初期化#############################################

        #グラフの枠を作っていく
        fig = plt.figure()
        ax = Axes3D(fig)

        #軸にラベルを付けたいときは書く
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        #点群を描画
        ax.plot(X,Y,Z,marker="o",linestyle='None',color="green")

        #法線を描画
        U, V, W = Disassemble(normals)
        ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True)

        #OBBを描画
        OBBViewer(ax, points)
        
        ###########################################################

        #result = figOptimize(points, normals, length, fig_type)
        result = figOptimize2(X, Y, Z, normals, length, fig_type)
        print(result)

        #fig_typeに応じた図形を選択
        if fig_type==0:
            figure = F.sphere(result.x)
        elif fig_type==1:
            figure = F.plane(result.x)

        #図形に対して"条件"を満たす点群を数える
        #label_list, X2, Y2, Z2= CountPoints(figure, points, X, Y, Z, normals, epsilon=0.01*length, alpha=np.pi)

        print("label_list:{}".format(label_list))

        #条件を満たす点群, 最適化された図形描画
        ax.plot(X2,Y2,Z2,marker=".",linestyle='None',color="orange")
        plot_implicit(ax, figure.f_rep, points, AABB_size=1, contourNum=15)

        #最後に.show()を書いてグラフ表示
        plt.show()

        score.append(result.fun)
        para.append(result.x)


    
    #スコアが最小の図形を描画
    best_fig = score.index(min(score))

    if best_fig==0:
        figure = F.sphere(para[best_fig])
    elif best_fig==1:
        figure = F.plane(para[best_fig])

    plot_implicit(ax, figure.f_rep, points, AABB_size=1, contourNum=15)
"""


OptiViewer("../data/pumpkin.obj", 0)
#DetectViewer("")