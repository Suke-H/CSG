import numpy as np
from graphviz import Digraph
import itertools
import copy

from ASTtest2 import AST
from GAtest import *

def Depth(i):
    return int(np.log2(i+1)+1)

def Adapt(i, k0):
    return int(2**(Depth(i)-1)*k0 + i)

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
            print(Adapt(i, link_point))
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

        print("point{}:depth{}に接続".format(mutate_point, Depth(mutate_point)))

        # つけ木作成
        branch = AST(branch_depth)
        branch.InitializeRandomPerson(["X1", "X2", "X3"])

        # つけ木ををっつける
        tree2 = LinkTree(tree, branch, mutate_point)

        DrawTree(tree, "img/mutate1")
        DrawTree(branch, "img/mutate2")
        DrawTree(tree2, "img/mutate3")

def Crossover(tree1, tree2, rate=1.0):
    # rateの確率で交叉
    if np.random.rand() <= rate:
        # 木をスキャン
        node_list1, _, _, _ = tree1.Scan()
        node_list2, _, _, _ = tree2.Scan()
        print(node_list1, node_list2)
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

        DrawTree(tree1, "img/cross1")
        DrawTree(tree2, "img/cross2")
        DrawTree(branch1, "img/cross3")
        DrawTree(branch2, "img/cross4")
        DrawTree(cross_tree1, "img/cross5")
        DrawTree(cross_tree2, "img/cross6")

"""       
tree = AST(4)
leaf_list = np.asarray(["X1", "X2", "X3"])
tree.InitializeRandomPerson(leaf_list)
Mutate(tree)
"""

tree1 = AST(4)
tree2 = AST(4)
leaf_list = np.asarray(["X1", "X2", "X3"])
tree1.InitializeRandomPerson(leaf_list)
tree2.InitializeRandomPerson(leaf_list)
Crossover(tree1, tree2)
