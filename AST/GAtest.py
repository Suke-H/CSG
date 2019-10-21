import numpy as np
import random
from graphviz import Digraph
import itertools

from ASTtest import AST

#ランダムにn人クリーチャー作成
def InitializeRandomPeople(n):
    People_list = []

    for i in range(n):
        #初期化
        p = AST(15)
        #leafの種類
        leaf_list = np.asarray(["X1", "X2", "X3"])
        #一人生成
        _, _, _, _ = p.InitializeRandomPerson(leaf_list)
        #リストにクラスごと登録
        People_list.append(p)

    return People_list

#Scoreの大きい順に並び変え
def Rank(People_list):
    score_list = []

    #リストにスコアを記録していく
    for p in People:
        score_list.append(p.Score())

    #Scoreの大きい順からインデックスを読み上げ、リストに記録
    index_list = sorted(range(len(score_list)), reverse=True, key=lambda k: score_list[k])

    #index_listの順にPeople_listを並べる
    return np.array(People_list)[index_list]

#突然変異
def Mutate(tree, rate=0.2):
    #rateの確率で突然変異
    if np.random.rand() <= rate:
        #木をスキャン
        node_num_list, node_key_list, L, edge_list = tree.Scan()
        
        #突然変異させる木の要素をランダム選択
        mutate_point = np.random.choice(node_num_list)



def DrawTree(tree, path):
    #木をスキャン
    node_num_list, node_key_list, L, edge_list = tree.Scan()

    # formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
    G = Digraph(format='png')
    G.attr('node', shape='circle')

    #二分木作成
    for num, key in zip(node_num_list, node_key_list):
        G.node(str(num), key)

    for i, j in edge_list:
        G.edge(str(i), str(j))

    G.render(path)


#10人作ってみる
People = InitializeRandomPeople(10)
#Score_list
Score_list = []
for p in People:
    Score_list.append(p.Score())
print(Score_list)

#Rank
sorted_People = Rank(People)
new_list = []
for p in sorted_People:
    new_list.append(p.Score())
print(new_list)

#一位を描画
DrawTree(sorted_People[0], "img/GAtest")
