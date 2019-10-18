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
from FigureDetection import CountPoints, MarkPoints, LabelPoints

#seabornはimportしておくだけでもmatplotlibのグラフがきれいになる
import seaborn as sns
sns.set_style("darkgrid")

def ViewerInit(points, X, Y, Z, normals):
    #グラフの枠を作っていく
    fig = plt.figure()
    ax = Axes3D(fig)

    #軸にラベルを付けたいときは書く
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    #点群を描画
    ax.plot(X,Y,Z,marker="o",linestyle='None',color="white")

    #法線を描画
    #U, V, W = Disassemble(normals)
    #ax.quiver(X, Y, Z, U, V, W,  length=0.1, normalize=True)

    #OBBを描画
    OBBViewer(ax, points)

    return ax

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

def DetectViewer(path):
    #点群,法線,OBBの対角線の長さ  取得
    #points, X, Y, Z, normals, length = PreProcess(path)
    
    #自作の点群を扱いたいときはこちら
    points, X, Y, Z, normals, length = PreProcess2()

    #元の点群データを保存しておく
    ori_points = points[:, :]

    scores = []
    paras = []
    indices = []

    fitting_figures = []
    
    print("points:{}".format(points.shape[0]))

    ###グラフ初期化###
    ax = ViewerInit(points, X, Y, Z, normals)

    while points.shape[0] >= ori_points.shape[0] * 0.1:
        ###最適化###
        for fig_type in [0, 1]:

            ###グラフ初期化##
            #ax = ViewerInit(points, X, Y, Z, normals)

            #図形フィッティング
            #result = figOptimize(points, normals, length, fig_type)
            result = figOptimize2(X, Y, Z, normals, length, fig_type)
            print(result.x)

            #fig_typeに応じた図形を選択
            if fig_type==0:
                figure = F.sphere(result.x)
            elif fig_type==1:
                figure = F.plane(result.x)

            #図形描画
            #plot_implicit(ax, figure.f_rep, points, AABB_size=1, contourNum=50)

            #図形に対して"条件"を満たす点群を数える、これをスコアとする
            MX, MY, MZ, num, index = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.04*length, alpha=np.pi/10)
            print("num:{}".format(num))

            #条件を満たす点群, 最適化された図形描画
            #ax.plot(MX,MY,MZ,marker=".",linestyle='None',color="orange")
            
            #最後に.show()を書いてグラフ表示
            #plt.show()

            #スコアとパラメータ,インデックスを保存
            scores.append(num)
            paras.append(result.x)
            indices.append(index)

        if sum(scores) <= 5:
            print("もっかい！\n")
            continue

        ###グラフ初期化###
        ax = ViewerInit(points, X, Y, Z, normals)

        #スコアが最大の図形を描画
        best_fig = scores.index(max(scores))

        if best_fig==0:
            figure = F.sphere(paras[best_fig])
            fitting_figures.append("球：[" + ','.join(map(str, list(paras[best_fig]))) + "]")
        elif best_fig==1:
            figure = F.plane(paras[best_fig])
            fitting_figures.append("平面：[" + ','.join(map(str, list(paras[best_fig]))) + "]")

        plot_implicit(ax, figure.f_rep, points, AABB_size=1, contourNum=15)

        #plt.show()

        #フィットした点群を削除
        points = np.delete(points, indices[best_fig], axis=0)
        normals = np.delete(normals, indices[best_fig], axis=0)
        X, Y, Z = Disassemble(points)
        

        ###グラフ初期化###
        ax = ViewerInit(points, X, Y, Z, normals)

        plt.show()
        ##################
        print("points:{}".format(points.shape[0]))
        

    print(len(fitting_figures), fitting_figures)
    plt.show()






#OptiViewer("../data/pumpkin.obj", 0)
DetectViewer("")