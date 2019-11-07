import numpy as np
import random
from graphviz import Digraph
import itertools
import copy

from ASTtest2 import AST

def GA(epoch=50, N=30, n=5, depth=4, tournament_size=5, cross_rate=0.75, mutate_rate=0.1):
    # N人クリーチャー作成
    people = InitializeRandomPeople(N, depth=4)

    for i in range(epoch):
        print("epoch:{}".format(i))
        # スコア順に並び替え
        people = Rank(people)
        # 上位n人は保存
        next_people = people[:n]

        # 次世代がN人超すまで
        # トーナメント選択->交叉、突然変異->保存
        # を繰り返す
        c = 0
        while len(next_people) <= N:
            #print("{} 回目".format(c))
            # トーナメントサイズの人数出場
            entry = np.random.choice(people, tournament_size, replace=False)
            # 上位3人選択
            entry = Rank(entry)[:3]
            # 1位と2位を交叉
            c1, c2 = Crossover(entry[0], entry[1], rate=cross_rate)
            # 3位を突然変異
            c3 = Mutate(entry[2], rate=mutate_rate)
            
            next_people = np.append(next_people, c1)
            next_people = np.append(next_people, c2)
            next_people = np.append(next_people, c3)

            c+=1

        people = next_people[:]

        if i % 10 == 0:
            #Rank
            sorted_People = Rank(people)
            new_list = []
            for p in sorted_People:
                new_list.append(p.Score())
            print(new_list)

            #一位を描画
            DrawTree(sorted_People[0], "img/GA"+str(int(i/10)))

    #Rank
    sorted_People = Rank(people)
    new_list = []
    for p in sorted_People:
        new_list.append(p.Score())
    print(new_list)

    #一位を描画
    DrawTree(sorted_People[0], "img/GAlast")

def Depth(i):
    return int(np.log2(i+1)+1)

def Adapt(i, k0):
    return int(2**(Depth(i)-1)*k0 + i)

#ランダムにn人クリーチャー作成
def InitializeRandomPeople(n, depth=4):
    People_list = []

    for i in range(n):
        #初期化
        p = AST(depth)
        #leafの種類
        leaf_list = np.asarray(["X1", "X2", "X3"])
        #一人生成
        p.InitializeRandomPerson(leaf_list)
        #リストにクラスごと登録
        People_list.append(p)

    return People_list

#Scoreの大きい順に並び変え
def Rank(People_list):
    #リストにスコアを記録していく
    score_list = [People_list[i].Score() for i in range(len(People_list))]

    #Scoreの大きい順からインデックスを読み上げ、リストに記録
    index_list = sorted(range(len(score_list)), reverse=True, key=lambda k: score_list[k])

    #index_listの順にPeople_listを並べる
    return np.array(People_list)[index_list]


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

# サイズエラーはならない体
def LinkTree(tree1, tree2, link_point):
    #print("Linking...")
    # tree1に直接書き込まないようコピー
    tree = copy.deepcopy(tree1)

    # 削除する(ために探索する)深さ
    delete_depth = tree.depth - Depth(link_point) + 1
    #print("depth:{}".format(delete_depth))
    # 探索して削除
    for i in range(2**(delete_depth)-1):
        if tree.ast[Adapt(i, link_point)]:
            #print(Adapt(i, link_point))
            tree.ast[Adapt(i, link_point)] = None
            tree.size = tree.size - 1

    node_num_list, node_key_list, _, _ = tree2.Scan()

    # 0はlink_point, 0以外は Adapt(i, link_point)
    node_num_list = [Adapt(node_num_list[i], link_point) if i!=0 else link_point \
                    for i in range(len(node_num_list))]

    # tree1にtree2をくっつける
    for i, key in zip(node_num_list, node_key_list):
        tree.ast[int(i)] = key
        tree.size = tree.size + 1

    return tree

def Mutate(tree, rate=1.0):
    # rateの確率で突然変異
    if np.random.rand() <= rate:
        # 木をスキャン
        node_num_list, _, _, _ = tree.Scan()
        # つけ木の深さ
        branch_depth = 0
        # 深さが1を超えるまで
        while branch_depth <= 0:
        
            # 突然変異させる木の要素をランダム選択
            mutate_point = int(np.random.choice(node_num_list))
            # つけ木の深さ　= 木の最大の深さ - mutate_pointの深さ + 1
            branch_depth = tree.depth - Depth(mutate_point) + 1

        #print("point{}:depth{}に接続".format(mutate_point, Depth(mutate_point)))

        # つけ木作成
        branch = AST(branch_depth)
        branch.InitializeRandomPerson(["X1", "X2", "X3"])

        # つけ木ををっつける
        mutate_tree = LinkTree(tree, branch, mutate_point)

        return mutate_tree

    # ハズレたら何もせずに返す    
    else:
        return tree

def Crossover(tree1, tree2, rate=1.0):
    # rateの確率で交叉
    if np.random.rand() <= rate:
        # 木をスキャン
        node_list1, _, _, _ = tree1.Scan()
        node_list2, _, _, _ = tree2.Scan()

        #print(node_list1, node_list2)
        # 交叉をした後の最大の深さ
        cross_depth1 = tree1.depth + 1
        cross_depth2 = tree2.depth + 1


        # 交叉する際にtree1と2それぞれで最大の深さを超えないように、point1と2を選択
        while cross_depth1 > tree1.depth or cross_depth2 > tree2.depth:

            # 交叉する木の要素をランダム選択
            point1 = int(np.random.choice(node_list1))
            point2 = int(np.random.choice(node_list2))

            ### 以後、pointをrootとした木を探索してつくる ###
            # それぞれの木を格納するリスト
            branch1_node_list = []
            branch1_key_list = []
            branch2_node_list = []
            branch2_key_list = []

            #探索する深さ
            search_depth1 = tree1.depth - Depth(point1) + 1
            #発見したノードの最大値
            max_node = 0
            # point以降のノードを探索してつけ木を作成
            for i in range(2**(search_depth1)-1):
                if tree1.ast[Adapt(i, point1)]:
                    branch1_node_list.append(i)
                    branch1_key_list.append(tree1.ast[Adapt(i, point1)])
                    max_node = i

            #branch1の深さ
            branch1_depth = Depth(max_node)

            #同じ処理
            search_depth2 = tree2.depth - Depth(point2) + 1
            max_node = 0
            for i in range(2**(search_depth2)-1):
                if tree2.ast[Adapt(i, point2)]:
                    branch2_node_list.append(i)
                    branch2_key_list.append(tree2.ast[Adapt(i, point2)])
                    max_node = i
            branch2_depth = Depth(max_node)

            # つけ木の深さ　= 木の最大の深さ - pointの深さ + 1
            branch1_depth = tree1.depth - Depth(point1) + 1
            branch2_depth = tree2.depth - Depth(point2) + 1

            # 交叉した場所の木の深さ = pointの深さ + つけ木の深さ - 1
            cross_depth1 = Depth(point1) + branch2_depth - 1
            cross_depth2 = Depth(point2) + branch1_depth - 1

        """
        print("tree1:{}, tree2:{}".format(tree1.depth, tree2.depth))
        print("point1:{}, depth{}, point2:{}, depth{}".format(point1, Depth(point1), point2, Depth(point2)))
        print("b1depth:{}, b2depth:{}".format(branch1_depth, branch2_depth))
        print("cross1:{}, cross2:{}".format(cross_depth1, cross_depth2))
        print(branch1_node_list, branch2_node_list)
        """

        # つけ木作成
        branch1 = AST(branch1_depth)
        branch2 = AST(branch1_depth)
        # 葉の候補数を0に(これ無くしたい)
        branch1.leaf_num = branch2.leaf_num = 0
        for i, key in zip(branch1_node_list, branch1_key_list):
            branch1.ast[i] = key
            branch1.size = branch1.size + 1
        for i, key in zip(branch2_node_list, branch2_key_list):
            branch2.ast[i] = key
            branch2.size = branch2.size + 1

        # つけ木をくっつける
        cross_tree1 = LinkTree(tree1, branch2, point1)
        cross_tree2 = LinkTree(tree2, branch1, point2)

        return cross_tree1, cross_tree2

    # ハズレたら何もせずに返す
    else:
        return tree1, tree2

"""
#10人作ってみる
People = InitializeRandomPeople(5, depth=2)
print(People)
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
"""

GA(epoch=50, N=30, n=5, depth=4, tournament_size=10, cross_rate=0.75, mutate_rate=0.1)