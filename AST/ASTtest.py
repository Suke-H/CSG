import numpy as np
import random
from graphviz import Digraph
import itertools

class AST:
    def __init__(self, depth):
        #木の深さ
        self.depth = depth
        #要素数
        self.size = 0
        #葉の候補数
        self.leaf_num = 1
        #ASTの配列
        self.ast = np.asarray([None for i in range(2**self.depth-1)])
        #後置記法リスト
        self.postorder_list = np.asarray([])

    #####AST生成用メソッド#########################################

    def InitializeRandomPerson(self, num, leaf_list):
        #key_list作成...ランダムにand,or,notを入れたもの
        key_list = np.asarray([])
        for i in range(num):
            key = np.random.choice(["and", "or", "not"], p=[0.35, 0.35, 0.3])
            key_list = np.append(key_list, key)

        #key_listをASTにランダム挿入
        for key in key_list:
            self.RandomInsert(key)

        #leaf_listをAStに挿入
        self.LeafRandomInsert(leaf_list)

        #描画用にAST全体をスキャン
        node_num_list, node_key_list, L, edge_list = self.Scan()

        return node_num_list, node_key_list, L, edge_list
        
    def RandomInsert(self, key):
        #randomInsertをするたびにsizeとleaf_numを増やす
        #Javaのprivateみたいにメンバ変数をいじらないようにする必要性あり？
        self.size = self.size + 1
        if key != "not":
            self.leaf_num = self.leaf_num + 1

        i = 0
        #rootに何もなければ挿入して終わり
        if self.ast[i] is None:
            self.ast[i] = key
            return        

        #Noneになるまで行く
        while self.ast[i]:
            #50%で左か右に行く(notなら左)
            if random.random() >= 0.5 or self.ast[i] == "not":
                i = 2*i+1
            else:
                i = 2*i+2

        self.ast[i] = key

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
        i = int(0)
        node_num_list = np.asarray([])
        node_key_list = np.asarray([])
        leaf_list = np.asarray([])

        #iをノードと葉がなくなるまで走査して、すべてのノードと葉をピック
        while len(node_num_list) < self.size or len(leaf_list) < self.leaf_num:
            #ast[i]があったらノード
            if self.ast[i]:
                node_num_list = np.append(node_num_list, i)
                node_key_list = np.append(node_key_list, self.ast[i])
            #ast[i]はない場合
            elif self.ast[(i-1)//2]:
                #iがnotの右の子でないなら葉
                if i % 2 == 1 or self.ast[abs((i-1)//2)] != "not":
                    leaf_list = np.append(leaf_list, i)

            i = i + 1

        #np.appendのときのおまじない
        edge_list = np.empty((0,2), int)

        if len(node_num_list) >= 2:
            #子の親(=(num-1)//2)は確実に存在するのを利用し、辺を結ぶ
            #print(node_num_list[1:])
            for num in node_num_list[1:]:
                #なぜか(1-1)//2=-0となり,0と別物になるのでabsを利用している
                edge_list = np.append(edge_list, np.asarray([[abs((num-1)//2), num]]), axis=0)

        return node_num_list, node_key_list, leaf_list, edge_list


    

    #####スコアメソッド#########################################

    #逆ポーランド記法でASTを表記
    def Postorder(self, i):
        if self.ast[i] is None:
            return

        self.Postorder(2*i+1)
        self.Postorder(2*i+2)

        self.postorder_list = np.append(self.postorder_list,self.ast[i])

    def Score(self):
        #postorder_listに逆ポーランド記法を書き込む
        self.Postorder(0)

        #print(self.postorder_list)

        operator = {
        'and': (lambda x, y: x and y),
        'or': (lambda x, y: x or y),
        'not': (lambda x: not x),
        }

        #各割当て(2^3通り)に対する真理値が、目標のブール関数と一致した数がスコア
        goal_truth = [False, False, False, True, False, True, True, True]
        bit = [False, True]
        #goal_truth = np.asarray([0,0,0,1,0,1,1,1])
        #bit = np.asarray([0,1])
        truth_assignment = list(itertools.product(bit, bit, bit))
        #print(truth_assignment)

        score = 0

        #各割当てでの真理値の一致チェック
        for assign, goal in zip(truth_assignment, goal_truth):
            X1 = assign[0]
            X2 = assign[1]
            X3 = assign[2]
            bool_list = []

            #逆ポーランド記法計算の下準備
            for key in self.postorder_list:
                #X1, X2, X3があったら割り当てを代入
                if key == "X1":
                    bool_list.append(X1)

                elif key == "X2":
                    bool_list.append(X2)

                elif key == "X3":
                    bool_list.append(X3)

                else:
                    bool_list.append(key)

            #print("bool_list:{}".format(bool_list))
            #print(bool_list)
            #print('RPN: {}'.format(bool_list))

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

            if stack[0] == goal:
                score = score + 1

        return score

"""
count = score = 0

while(score != 8):

    test = AST(15)
    leaf_list = np.asarray(["X1", "X2", "X3"])
    node_num_list, node_key_list, L, edge_list = test.InitializeRandomPerson(5 , leaf_list)
    #print(node_num_list, node_key_list, L, edge_list)
    score = test.Score()
    print(count, score)
    count = count + 1
    

# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='png')
G.attr('node', shape='circle')

#二分木作成
for num, key in zip(node_num_list, node_key_list):
    G.node(str(num), key)

for i, j in edge_list:
    G.edge(str(i), str(j))

G.render("ASTtest2")


print(test.size, test.leaf_num)
node_num_list, node_key_list, leaf_list, edge_list = test.scan()
print(node_num_list, node_key_list, leaf_list, edge_list)
print(test.ast[0:30])

"""