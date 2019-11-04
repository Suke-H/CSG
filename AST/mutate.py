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
    print("Linking...")
    # tree1に直接書き込まないようコピー
    tree = copy.deepcopy(tree1)

    # 削除する深さ
    delete_depth = tree.depth - Depth(link_point) + 1
    print("depth:{}".format(delete_depth))
    # 逐次的に削除
    for i in range(2**(delete_depth)-1):
        if tree.ast[Adapt(i, link_point)]:
            print(Adapt(i, link_point))
            tree.ast[Adapt(i, link_point)] = None
            tree.size = tree.size - 1

    node_num_list, node_key_list, _, _ = tree2.Scan()
    print(node_num_list)

    # 0はlink_point, 0以外は 2**(Depth(i)-1)*link_point + i
    node_num_list = [Adapt(node_num_list[i], link_point) if i!=0 else link_point \
                    for i in range(len(node_num_list))]
    print(node_num_list)

    # tree1にtree2をくっつける
    for i, key in zip(node_num_list, node_key_list):
        print(int(i), key)
        tree.ast[int(i)] = key
        tree.size = tree.size + 1

    return tree

def mutate(tree, rate=1.0):
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

        # つけ木をくっつける
        tree2 = LinkTree(tree, branch, mutate_point)

        DrawTree(tree, "img/mutate1")
        DrawTree(branch, "img/mutate2")
        DrawTree(tree2, "img/mutate3")



tree = AST(4)
leaf_list = np.asarray(["X1", "X2", "X3"])
tree.InitializeRandomPerson(leaf_list)
mutate(tree)

