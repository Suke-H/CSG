import numpy as np
import random

node_list = np.asarray([])

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class AST:
    def __init__(self):
        self.root = None

    def randomInsert(self, key):
        print(key)
        pwd = self.root

        if pwd is None:
            self.root = Node(key)
            print("<END>")
            return

        #行く場所がなくなる(左も右もNoneになる)まで行く
        while pwd.right or pwd.left:

            tmp = pwd

            #右に行けなかったら左に(notのとき左に一方通行)
            if pwd.right is None or pwd.key == "not":
                pwd = pwd.left

            #左に行けなかったら右に
            elif pwd.left is None:
                pwd = pwd.right

            #どっちにも行けるなら50%で左か右に行く
            elif random.random() >= 0.5 :
                pwd = pwd.left 

            else:
                pwd = pwd.right

            print(tmp.key + "→" + pwd.key)

        #50%の確率で左か右に挿入(notなら左)
        if random.random() >= 0.5 or pwd.key == "not":
            print("<ENDA>")
            pwd.left = Node(key)

        else:
            print("<ENDB>")
            pwd.right = Node(key)


    def preorder(self, pwd):
        if pwd is None:
            return

        print(pwd.key)
        node_list = np.append(node_list, pwd.key)

        self.preorder(pwd.left)
        self.preorder(pwd.right)


    def drawAST(self):
        pwd = self.root

        if pwd is None:
            self.root = Node(key)
            return

        while

    """
    リストバージョン
    def randomInsert(self, key):
        #randomInsertをするたびにsizeとleaf_numを増やす
        #Javaのprivateみたいにメンバ変数をいじらないようにする必要性あり？
        self.size = self.size + 1
        if key != "not":
            self.leaf_num = self.leaf_num + 1


        print(key)
        i = 0

        #rootに何もなければ挿入して終わり
        if self.ast[i] is None:
            self.ast[i] = key
            print("<END>")
            return        

        #行く場所がなくなる(左も右もNoneになる)まで行く
        while self.ast[2*i+1] or self.ast[2*i+2]:

            tmp = i

            #右に行けなかったら左に(notのとき左に一方通行)
            if self.ast[2*i+2] is None or self.ast[i] == "not":
                i = 2*i+1

            #左に行けなかったら右に
            elif self.ast[2*i+1] is None:
                i = 2*i+2

            #どっちにも行けるなら50%で左か右に行く
            elif random.random() >= 0.5 :
                i = 2*i+1

            else:
                i = 2*i+2

            print(self.ast[tmp] + "→" + self.ast[i])

        #50%の確率で左か右に挿入(notなら左)
        if random.random() >= 0.5 or self.ast[i] == "not":
            print("<ENDA>")
            self.ast[2*i+1] = key

        else:
            print("<ENDB>")
            self.ast[2*i+2] = key

    """


"""
test = AST()
test.randomInsert("and")
test.randomInsert("not")
test.randomInsert("or")
test.randomInsert("and")

#test.root = Node("and")
#test.root.left = Node("not")
#test.root.right = Node("or")
#test.root.left.left = Node("aaa")

test.randomInsert("yeah")
#test.randomInsert("Hello")
print("="*50)
#print(test.root.key)
test.preorder(test.root)
print(node_list)
"""
