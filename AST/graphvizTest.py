from graphviz import Digraph

#from GAtest import Node, AST


# formatはpngを指定(他にはPDF, PNG, SVGなどが指定可)
G = Digraph(format='png')
G.attr('node', shape='circle')

"""
test = AST()
test.randomInsert("and")
test.randomInsert("not")
test.randomInsert("or")
test.randomInsert("and")
test.randomInsert("yeah")

test.preorder(test.root)

from GAtest import node_list

print(node_list)

"""
G.node("0", "and")
G.node("1", "and")
G.edge("0", "1")
# binary_tree.pngで保存
G.render('AST')
