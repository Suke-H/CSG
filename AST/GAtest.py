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
        _, _, _, _ = p.InitializeRandomPerson(5 , leaf_list)
        #リストにクラスごと登録
        People_list.append(p)

    return People_list

def Rank(People_list):
    Score_list = []

    #リストにスコアを記録していく
    for p in People:
        Score_list.append(p.Score())

    #Scoreの大きい順からインデックスを読み上げ、リストに記録
    index_list = sorted(range(len(Score_list)), reverse=True, key=lambda k: Score_list[k])

    #index_listの順にPeople_listを並べる
    return np.array(People_list)[index_list]


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