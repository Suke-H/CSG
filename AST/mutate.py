import numpy as np
from graphviz import Digraph
import itertools

from ASTtest2 import AST
from GAtest import *

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
            mutate_point = np.random.choice(node_num_list)
            # つけ木の深さ　= 木の最大の深さ - mutate_pointの深さ
            branch_depth = tree.depth - int(np.log2(mutate_point+1) + 1)

        print(mutate_point, branch_depth, tree.depth)

        # つけ木作成
        branch = AST(branch_depth)
        branch.InitializeRandomPerson(["X1", "X2", "X3"])

        DrawTree(branch, "img/mutate")



tree = AST(6)
leaf_list = np.asarray(["X1", "X2", "X3"])
tree.InitializeRandomPerson(leaf_list)
mutate(tree)

