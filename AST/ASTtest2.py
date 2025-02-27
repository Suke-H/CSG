import numpy as np
import random
from graphviz import Digraph
import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#他ディレクトリからのインポート
import sys
sys.path.append('../main/')
from PreProcess2 import PreProcess2
from FigureDetection import CountPoints
import figure2 as F
from figure_sample import *
from method import *

class AST:
    def __init__(self, depth):
        #木の深さ
        self.depth = depth
        #最大要素数(深さ"depth"での完全二分木の要素数)
        self.max_size = 2**self.depth-1
        #ASTの配列
        self.ast = np.array([None for i in range(self.max_size)])

        ###この2つはできれば使わないようにするプログラムにしたい##
        #要素数
        self.size = 0
        #葉の候補数
        self.leaf_num = 1

        #後置記法リスト
        self.postorder_list = np.asarray([])

    #####AST生成用メソッド#########################################

    def InitializeRandomPerson(self, leaf_list):
        #葉以外の要素数をランダムで決める(最大は2**(depth-1)-1)
        num = np.random.randint(0, 2**(self.depth-1))

        #key_list作成...ランダムにand,or,notを入れたもの
        key_list = [np.random.choice(["and", "or", "not"], p=[0.35, 0.35, 0.3]) for i in range(num)]

        # key_listをASTにランダム挿入
        # (num=0のときはkey_listが空になり一回もInsertされない)
        for key in key_list:
            self.RandomInsert(key)

        #leaf_listをAStに挿入
        # (num=0のときはroot1つにkeyが挿入される)
        self.LeafRandomInsert(leaf_list)
        
    #非終端記号keyを挿入
    #この後,葉を挿入するので、最大の深さ(self.depth)を超えないようにする
    def RandomInsert(self, key):
        #木が空のとき
        if self.size == 0:
            index = 0

        else:
            # leaf_listを取得
            _, _, leaf_list, _ = self.Scan()
            # 最大の深さにある場所を選択しないようにする
            leaf_list = leaf_list[np.where(np.log2(leaf_list+1)//1 + 1 < self.depth)]
            #print(leaf_list)

            # 置くとこないなら終わり(notが上に配置された時に起こりやすい)
            if len(leaf_list) == 0:
                ###テスト###
                #print(self.size, -1, key)
                return
            
            #leaf_listからランダムで場所を選択
            index = int(np.random.choice(leaf_list))
        ###テスト###   
        #print(self.size, index, key)
        self.ast[index] = key

        #randomInsertをするたびにsizeとleaf_numを増やす
        #Javaのprivateみたいにメンバ変数をいじらないようにする必要性あり？
        self.size = self.size + 1
        if key != "not":
            self.leaf_num = self.leaf_num + 1


    #終端記号を挿入
    def LeafRandomInsert(self, leaf_list):
        #葉の場所をスキャンで調べる
        _, _, leaf_num_list, _ = self.Scan()

        #ランダムに葉(=図形)を挿入
        for i in leaf_num_list:
            self.ast[int(i)] = np.random.choice(leaf_list)
            #挿入するたびsizeをプラス
            self.size = self.size + 1

        #葉の候補数を0に
        self.leaf_num = 0

    ###次の４つのリストを返す######################################
    #node_num_list:ASTの要素がある場所を格納[0,1,2,3,5,...]
    #node_key_list:↑に対応するkeyを格納[and,or,or,not,and,...]
    #leaf_list:ASTの葉を挿入すべき場所を格納[12,13,..]
    #edge_list:ASTの要素の辺を格納[[0,1],[0,2],...]
    #############################################################

    def Scan(self):
        node_num_list = np.asarray([])
        node_key_list = np.asarray([])
        leaf_list = np.asarray([])
        # np.appendのときのおまじない
        edge_list = np.empty((0,2), int)

        #print(self.size, self.leaf_num)

        # 空のとき(leaf_listだけ[0])
        if self.size==0:
            return node_num_list, node_key_list, np.append(leaf_list, 0), edge_list

        i = 0
        # iで木の全要素を走査して、すべてのノードと葉をピック
        while len(node_num_list) < self.size or len(leaf_list) < self.leaf_num:
            # ast[i]があったらノード
            if self.ast[i]:
                node_num_list = np.append(node_num_list, i)
                node_key_list = np.append(node_key_list, self.ast[i])
            # ast[i]はない場合
            elif self.ast[(i-1)//2]:
                #iがnotの右の子でないなら葉
                if i % 2 == 1 or self.ast[abs((i-1)//2)] != "not":
                    leaf_list = np.append(leaf_list, i)

            i = i + 1

        if len(node_num_list) >= 2:
            # 子の親(=(num-1)//2)は確実に存在するのを利用し、辺を結ぶ
            #print(node_num_list[1:])
            for num in node_num_list[1:]:
                # なぜか(1-1)//2=-0となり,0と別物になるのでabsを利用している
                edge_list = np.append(edge_list, np.asarray([[abs((num-1)//2), num]]), axis=0)

        return  node_num_list, node_key_list, leaf_list, edge_list
    
    #####スコアメソッド#########################################

    # 逆ポーランド記法でASTを表記
    def Postorder(self, i):
        # 今の場所が None または 配列のサイズを超してたら 終了
        if i >= self.max_size:
            return
        elif self.ast[i] is None:
            return
        
        # 左へ
        self.Postorder(2*i+1)
        # 右へ
        self.Postorder(2*i+2)
        # 代入
        self.postorder_list = np.append(self.postorder_list,self.ast[i])

    def BoolScore(self):
        # postorder_listに逆ポーランド記法を書き込む
        self.postorder_list = []
        self.Postorder(0)

        # 演算の定義
        operator = {
        'and': (lambda x, y: x and y),
        'or': (lambda x, y: x or y),
        'not': (lambda x: not x),
        }

        # 各割当て(2^3通り)に対する真理値が、目標のブール関数と一致した数がスコア
        goal_truth = [False, False,  False, True, False, True, True, True]
        bit = [False, True]
        truth_assignment = list(itertools.product(bit, bit, bit))
        score = 0

        #各割当てでの真理値の一致チェック
        for assign, goal in zip(truth_assignment, goal_truth):
            X1 = assign[0]
            X2 = assign[1]
            X3 = assign[2]

            bool_list = []

            # 逆ポーランド記法計算の下準備
            for key in self.postorder_list:
                # X1, X2, X3があったら割り当てを代入
                if key == "X1":
                    bool_list.append(X1)

                elif key == "X2":
                    bool_list.append(X2)

                elif key == "X3":
                    bool_list.append(X3)

                else:
                    bool_list.append(key)

            #print("bool_list:{}".format(bool_list))
            #逆ポーランド記法の計算にスタックを利用
            #演算子が出るまでスタックし続け、演算子が出たらpop(and,orなら2つ、notなら1つ)
            #して計算結果をスタックする
            stack = []

            for index, z  in enumerate(bool_list):
                #if index > 0:
                    #print(stack)

                #演算子じゃなかったらスタック
                if z not in operator.keys():
                    stack.append(z)
                    continue

                #not
                elif z == "not":
                    x = stack.pop()
                    #print("not {} = {}".format(x, operator[z](x)))
                    stack.append(operator[z](x))

                #and, or
                else:
                    y = stack.pop()
                    x = stack.pop()
                    stack.append(operator[z](x, y))
                    #print('{} {} {} = {}'.format(x, z, y, operator[z](x, y)))

            #print("="*50)
            #print("{}, {}, {} -> {} == {}".format(X1, X2, X3, stack[0], goal))
            #print(len(stack))

            if stack[0] == goal:
                score = score + 1

        return score

    def Score(self, drawfig=False):
        # postorder_listに逆ポーランド記法を書き込む
        self.postorder_list = []
        self.Postorder(0)

        # 演算の定義
        operator = {
        'and': (lambda p1, p2: F.AND(p1, p2)),
        'or': (lambda p1, p2: F.OR(p1, p2)),
        'not': (lambda p: F.NOT(p))
        }

        stack = []

        for z in self.postorder_list:
            #演算子じゃなかったらスタック
            if z not in operator.keys():
                stack.append(z)
                continue

            #not
            elif z == "not":
                x = stack.pop()
                #print("not {} = {}".format(x, operator[z](x)))
                stack.append(operator[z](x))

            #and, or
            else:
                y = stack.pop()
                x = stack.pop()
                stack.append(operator[z](x, y))
                #print('{} {} {} = {}'.format(x, z, y, operator[z](x, y)))

        # 構文木によって演算された図形がstackに残る
        figure = stack[0]

        # フィットした点の数がスコア
        points, X, Y, Z, normals, length = PreProcess2()
        MX, MY, MZ, score, _ = CountPoints(figure, points, X, Y, Z, normals, epsilon=0.03*length, alpha=np.pi)

        # グラフ作成したいとき
        if drawfig:
            DrawFig(figure, X, Y, Z, MX, MY, MZ)

        return score

        #print("="*50)
        #print("{}, {}, {} -> {} == {}".format(X1, X2, X3, stack[0], goal))

def DrawFig(figure, X, Y, Z, MX, MY, MZ):
        # グラフの枠を作っていく
        fig = plt.figure()
        ax = Axes3D(fig)

        # 軸にラベルを付けたいときは書く
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # 点群を描画
        ax.plot(X,Y,Z,marker="o",linestyle='None',color="white")
        ax.plot(MX,MY,MZ,marker=".",linestyle='None',color="red")

        # 図形を描画
        plot_implicit(ax, figure.f_rep)

        plt.show()

"""
test = AST(5)
#leaf_list = np.asarray(["X1", "X2", "X3"])
leaf_list = np.array([P_Z0, P_Z1, P_X0, P_X1, P_Y0, P_Y1])
test.InitializeRandomPerson(leaf_list)

node_num_list, node_key_list, leaf_list, edge_list = test.Scan()

# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='png')
G.attr('node', shape='circle')

#二分木作成
for num, key in zip(node_num_list, node_key_list):
    if key == P_Z0:
        key = "P1"
    elif key == P_Z1:
        key = "P2"
    elif key == P_X0:
        key = "P3"
    elif key == P_X1:
        key = "P4"
    elif key == P_Y0:
        key = "P5"
    elif key == P_Y1:
        key = "P6"
    G.node(str(num), key)

for i, j in edge_list:
    G.edge(str(i), str(j))

G.render("img/score")

print("Score:{}".format(test.Score()))
"""